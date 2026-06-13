"""
Simulates a live stream of transaction events and publishes them to the
Kafka topic `transactions`.

Each event:
{
  "tx_id": "uuid",
  "user_id": "user_123",
  "amount": 245.50,
  "timestamp": "2026-06-13T10:15:30Z",
  "merchant_risk": 0.12,
  "is_foreign": 0,
  "device_change": 0
}
"""

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("TRANSACTIONS_TOPIC", "transactions")
EVENTS_PER_SEC = float(os.environ.get("EVENTS_PER_SEC", "5"))
N_USERS = int(os.environ.get("N_USERS", "200"))

# Small pool of "high risk" users to make the stream interesting
HIGH_RISK_USERS = {f"user_{i}" for i in range(0, N_USERS, 17)}


def make_producer(retries=20, delay=3):
    last_err = None
    for _ in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[producer] Kafka not ready yet ({e}); retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Kafka: {last_err}")


def generate_event():
    user_id = f"user_{random.randint(0, N_USERS - 1)}"
    high_risk = user_id in HIGH_RISK_USERS

    base_amount = random.lognormvariate(3.5, 1.2)
    if high_risk and random.random() < 0.3:
        base_amount *= random.uniform(3, 8)  # occasional spike

    merchant_risk = (
        random.betavariate(4, 4) if high_risk else random.betavariate(1.5, 8)
    )
    is_foreign = 1 if (high_risk and random.random() < 0.25) else (
        1 if random.random() < 0.03 else 0
    )
    device_change = 1 if (high_risk and random.random() < 0.2) else (
        1 if random.random() < 0.02 else 0
    )

    return {
        "tx_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": round(base_amount, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merchant_risk": round(merchant_risk, 4),
        "is_foreign": is_foreign,
        "device_change": device_change,
    }


def main():
    producer = make_producer()
    print(f"[producer] Connected to Kafka at {KAFKA_BOOTSTRAP}, topic='{TOPIC}'")
    print(f"[producer] Emitting ~{EVENTS_PER_SEC} events/sec across {N_USERS} users")

    interval = 1.0 / EVENTS_PER_SEC
    count = 0
    while True:
        event = generate_event()
        producer.send(TOPIC, key=event["user_id"], value=event)
        count += 1
        if count % 50 == 0:
            producer.flush()
            print(f"[producer] sent {count} events")
        time.sleep(interval)


if __name__ == "__main__":
    main()
