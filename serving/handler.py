"""
Custom TorchServe handler for the fraud-detection MLP.

Expects JSON input:
{
  "features": [amount, hour, tx_1h, tx_24h, avg_amt_24h, merchant_risk, is_foreign, device_change]
}

Returns:
{
  "fraud_probability": 0.0123,
  "is_fraud": false
}
"""

import json
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

FEATURE_DIM = 8
THRESHOLD = 0.5


class FraudModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        model_file = model_dir + "/fraud_model.pt"
        self.model = torch.jit.load(model_file, map_location=self.device)
        self.model.eval()
        self.initialized = True
        logger.info("Fraud model loaded successfully on %s", self.device)

    def preprocess(self, data):
        batch = []
        for row in data:
            body = row.get("body") or row.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            if isinstance(body, str):
                body = json.loads(body)

            features = body.get("features")
            if features is None or len(features) != FEATURE_DIM:
                raise ValueError(
                    f"'features' must be a list of {FEATURE_DIM} floats, got: {features}"
                )
            batch.append(features)

        tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
        return tensor

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            probs = self.model(data)
        return probs

    def postprocess(self, data):
        results = []
        for p in data.tolist():
            results.append(
                {
                    "fraud_probability": round(float(p), 6),
                    "is_fraud": bool(p > THRESHOLD),
                }
            )
        return results
