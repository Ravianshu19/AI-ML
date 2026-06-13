"""
Consumer service:

  1. Reads transaction events from Kafka topic `transactions`
  2. Maintains rolling per-user features in Redis (tx_count_1h, tx_count_24h,
     avg_amount_24h) using Redis sorted sets
  3. Builds the 8-dim feature vector expected by the fraud model
  4. Calls the TorchServe REST endpoint for inference
  5. Writes the scored result to Postgres (table: scores)
  6. Publishes the result to Kafka topic `scores`
  7. Records per-stage latency (feature build, inference, total)
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

import redis
import requests
import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer, KafkaProducer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TRANSACTIONS_TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")
SCORES_TOPIC = os.environ.get("SCORES_TOPIC", "scores")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

TORCHSERVE_URL = os.environ.get(
    "TORCHSERVE_URL", "http://localhost:8080/predictions/fraud"
)

PG_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=pipeline user=pipeline password=pipeline host=localhost port=5432",
)

WINDOW_1H = 3600
WINDOW_24H = 86400


# ---------------------------------------------------------------------------
# Connection helpers with retry (services may not be up yet under compose)
# ---------------------------------------------------------------------------
def connect_with_retry(name, factory, retries=30, delay=3):
    last_err = None
    for _ in range(retries):
        try:
            return factory()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[consumer] {name} not ready ({e}); retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to {name}: {last_err}")


def get_kafka_consumer():
    return KafkaConsumer(
        TRANSACTIONS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="fraud-scoring-consumer",
    )


def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def get_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    return r


def get_pg_conn():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                id SERIAL PRIMARY KEY,
                tx_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                tx_timestamp TIMESTAMPTZ NOT NULL,
                fraud_probability DOUBLE PRECISION NOT NULL,
                is_fraud BOOLEAN NOT NULL,
                tx_count_1h INT NOT NULL,
                tx_count_24h INT NOT NULL,
                avg_amount_24h DOUBLE PRECISION NOT NULL,
                latency_feature_ms DOUBLE PRECISION NOT NULL,
                latency_inference_ms DOUBLE PRECISION NOT NULL,
                latency_total_ms DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_scores_created_at ON scores (created_at);"
        )


# ---------------------------------------------------------------------------
# Feature engineering using Redis sorted sets
# ---------------------------------------------------------------------------
def compute_rolling_features(r: redis.Redis, user_id: str, amount: float, ts: float):
    """
    Maintains two Redis sorted sets per user:
      tx:{user_id}        -> member = unique id, score = timestamp (for counts)
      amt:{user_id}       -> member = "<id>:<amount>", score = timestamp (for avg)

    Returns (tx_count_1h, tx_count_24h, avg_amount_24h) computed BEFORE
    inserting the current event (i.e., based on history).
    """
    tx_key = f"tx:{user_id}"
    amt_key = f"amt:{user_id}"

    cutoff_1h = ts - WINDOW_1H
    cutoff_24h = ts - WINDOW_24H

    pipe = r.pipeline()
    pipe.zremrangebyscore(tx_key, 0, cutoff_24h)
    pipe.zremrangebyscore(amt_key, 0, cutoff_24h)
    pipe.zcount(tx_key, cutoff_1h, ts)
    pipe.zcount(tx_key, cutoff_24h, ts)
    pipe.zrangebyscore(amt_key, cutoff_24h, ts)
    results = pipe.execute()

    tx_count_1h = int(results[2])
    tx_count_24h = int(results[3])
    amt_members = results[4]

    if amt_members:
        amounts = [float(m.split(":")[-1]) for m in amt_members]
        avg_amount_24h = sum(amounts) / len(amounts)
    else:
        avg_amount_24h = 0.0

    # Insert current event for future lookups
    member_id = str(uuid.uuid4())
    pipe2 = r.pipeline()
    pipe2.zadd(tx_key, {member_id: ts})
    pipe2.zadd(amt_key, {f"{member_id}:{amount}": ts})
    pipe2.expire(tx_key, WINDOW_24H + 60)
    pipe2.expire(amt_key, WINDOW_24H + 60)
    pipe2.execute()

    return tx_count_1h, tx_count_24h, avg_amount_24h


def build_feature_vector(event: dict, tx_1h: int, tx_24h: int, avg_amt_24h: float):
    ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    hour = ts.hour
    return [
        event["amount"] / 1000.0,
        hour / 24.0,
        tx_1h / 10.0,
        tx_24h / 50.0,
        avg_amt_24h / 1000.0,
        event["merchant_risk"],
        float(event["is_foreign"]),
        float(event["device_change"]),
    ]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    print("[consumer] starting up...")
    consumer = connect_with_retry("Kafka consumer", get_kafka_consumer)
    producer = connect_with_retry("Kafka producer", get_kafka_producer)
    r = connect_with_retry("Redis", get_redis)
    conn = connect_with_retry("Postgres", get_pg_conn)
    ensure_table(conn)

    print("[consumer] ready. Consuming from topic:", TRANSACTIONS_TOPIC)

    insert_sql = """
        INSERT INTO scores (
            tx_id, user_id, amount, tx_timestamp, fraud_probability, is_fraud,
            tx_count_1h, tx_count_24h, avg_amount_24h,
            latency_feature_ms, latency_inference_ms, latency_total_ms
        ) VALUES %s
    """

    buffer = []
    last_flush = time.time()

    for msg in consumer:
        t_start = time.time()
        event = msg.value

        try:
            ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        ts_epoch = ts.timestamp()

        # --- Feature engineering ---
        t0 = time.time()
        tx_1h, tx_24h, avg_amt_24h = compute_rolling_features(
            r, event["user_id"], event["amount"], ts_epoch
        )
        features = build_feature_vector(event, tx_1h, tx_24h, avg_amt_24h)
        t1 = time.time()
        latency_feature_ms = (t1 - t0) * 1000

        # --- Inference ---
        try:
            resp = requests.post(
                TORCHSERVE_URL, json={"features": features}, timeout=5
            )
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list):
                result = result[0]
            fraud_prob = float(result["fraud_probability"])
            is_fraud = bool(result["is_fraud"])
        except Exception as e:  # noqa: BLE001
            print(f"[consumer] inference error: {e}")
            fraud_prob, is_fraud = -1.0, False
        t2 = time.time()
        latency_inference_ms = (t2 - t1) * 1000
        latency_total_ms = (t2 - t_start) * 1000

        row = (
            event["tx_id"],
            event["user_id"],
            event["amount"],
            ts,
            fraud_prob,
            is_fraud,
            tx_1h,
            tx_24h,
            avg_amt_24h,
            latency_feature_ms,
            latency_inference_ms,
            latency_total_ms,
        )
        buffer.append(row)

        # publish score event
        producer.send(
            SCORES_TOPIC,
            {
                "tx_id": event["tx_id"],
                "user_id": event["user_id"],
                "fraud_probability": fraud_prob,
                "is_fraud": is_fraud,
                "latency_total_ms": latency_total_ms,
            },
        )

        # batch insert into Postgres every ~1s or 25 rows
        if len(buffer) >= 25 or (time.time() - last_flush) > 1.0:
            try:
                with conn.cursor() as cur:
                    execute_values(cur, insert_sql, buffer)
                buffer.clear()
                last_flush = time.time()
            except Exception as e:  # noqa: BLE001
                print(f"[consumer] postgres insert error: {e}")
                conn = connect_with_retry("Postgres", get_pg_conn)
                ensure_table(conn)


if __name__ == "__main__":
    main()
