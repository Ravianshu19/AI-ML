"""
Train a lightweight fraud-detection MLP on synthetic transaction data
and export it as TorchScript for TorchServe.

Features (8):
  0: amount (scaled)
  1: hour_of_day (0-23, scaled)
  2: tx_count_last_1h   (rolling feature, computed at inference by consumer)
  3: tx_count_last_24h  (rolling feature)
  4: avg_amount_last_24h (rolling feature)
  5: merchant_risk_score (0-1)
  6: is_foreign (0/1)
  7: device_change (0/1)
"""

import numpy as np
import torch
import torch.nn as nn
import os

torch.manual_seed(42)
np.random.seed(42)

N = 50_000
FEATURE_DIM = 8


def generate_synthetic_data(n=N):
    amount = np.random.lognormal(mean=3.5, sigma=1.2, size=n)
    hour = np.random.randint(0, 24, size=n)
    tx_1h = np.random.poisson(1.0, size=n)
    tx_24h = np.random.poisson(8.0, size=n)
    avg_amt_24h = np.random.lognormal(mean=3.0, sigma=1.0, size=n)
    merchant_risk = np.random.beta(1.5, 8, size=n)
    is_foreign = np.random.binomial(1, 0.08, size=n)
    device_change = np.random.binomial(1, 0.05, size=n)

    # Synthetic fraud label: combination of risk factors + nonlinear interactions
    risk_score = (
        0.35 * (amount > 800).astype(float)
        + 0.25 * merchant_risk
        + 0.20 * is_foreign
        + 0.15 * device_change
        + 0.10 * (tx_1h >= 3).astype(float)
        + 0.10 * ((hour >= 1) & (hour <= 4)).astype(float)
        + 0.05 * (amount > 5 * (avg_amt_24h + 1e-6)).astype(float)
    )
    noise = np.random.normal(0, 0.08, size=n)
    prob_fraud = 1 / (1 + np.exp(-(6 * (risk_score + noise - 0.5))))
    label = np.random.binomial(1, prob_fraud)

    X = np.stack(
        [
            amount / 1000.0,
            hour / 24.0,
            tx_1h / 10.0,
            tx_24h / 50.0,
            avg_amt_24h / 1000.0,
            merchant_risk,
            is_foreign.astype(float),
            device_change.astype(float),
        ],
        axis=1,
    ).astype(np.float32)

    y = label.astype(np.float32)
    return X, y


class FraudMLP(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train():
    X, y = generate_synthetic_data()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    split = int(0.8 * len(X_t))
    X_train, X_val = X_t[:split], X_t[split:]
    y_train, y_val = y_t[:split], y_t[split:]

    model = FraudMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    epochs = 15
    batch_size = 512
    n_train = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5).float() == y_val).float().mean().item()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={total_loss/n_train:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    # Export TorchScript
    model.eval()
    example = torch.randn(1, FEATURE_DIM)
    traced = torch.jit.trace(model, example)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fraud_model.pt")
    traced.save(out_path)
    print(f"\nSaved TorchScript model to {out_path}")


if __name__ == "__main__":
    train()
