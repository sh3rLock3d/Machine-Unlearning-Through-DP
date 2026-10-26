"""Shared pieces for all experiments: data, splits, model, training, evaluation, cost tracking."""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_RATE_PER_HOUR = 0.40  # USD per GPU-hour -- set this to your GPU's actual rate
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def load_mnist(root="data"):
    """Load MNIST once and keep it on the GPU as normalized tensors (no DataLoader -> much faster)."""
    def prep(ds):
        x = ds.data.float().div(255).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
        return x.to(DEVICE), ds.targets.clone().to(DEVICE)

    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)
    return prep(train), prep(test)


def make_splits(y_train, path="splits.json", seed=0, forget_size=600):
    """Fix the forget set and the canary once, so every method uses the same ones.

    - forget: random 1% of the training set (for accuracy / MIA metrics)
    - canary: one training point with a flipped label (the 'hard point' for empirical epsilon)
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_train))
    canary_idx = int(perm[0])
    true_label = int(y_train[canary_idx])
    splits = {
        "canary_idx": canary_idx,
        "canary_true_label": true_label,
        "canary_label": int((true_label + rng.integers(1, 10)) % 10),  # guaranteed != true label
        "forget_idx": perm[1:1 + forget_size].tolist(),
    }
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits


def apply_canary(y_train, splits):
    """Flip the canary's label in place. Every model that trains on the canary sees the wrong label."""
    y_train[splits["canary_idx"]] = splits["canary_label"]
    return y_train


# ---------------------------------------------------------------- model
class SmallCNN(nn.Module):
    """~99% on MNIST. No BatchNorm, so the same model works with Opacus DP-SGD later."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ---------------------------------------------------------------- train / eval
def train(model, x, y, idx, epochs=5, batch_size=128, lr=1e-3):
    """Standard (non-private) training on the samples in idx."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), device=idx.device)]
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def per_sample_loss(model, x, y, batch_size=4096):
    """Per-example losses -- used for accuracy, MIA, and the empirical-epsilon statistic."""
    model.eval()
    losses, correct = [], []
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        losses.append(F.cross_entropy(out, y[i:i + batch_size], reduction="none"))
        correct.append(out.argmax(1) == y[i:i + batch_size])
    return torch.cat(losses), torch.cat(correct)


def accuracy(model, x, y):
    return per_sample_loss(model, x, y)[1].float().mean().item()


# ---------------------------------------------------------------- cost tracking
class CostTracker:
    """Wall-clock time (with CUDA sync), peak GPU memory, GPU-hours and $ for a block of code."""

    def __enter__(self):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0.0
        self.gpu_hours = self.seconds / 3600
        self.usd = self.gpu_hours * GPU_RATE_PER_HOUR if DEVICE.type == "cuda" else 0.0


def device_name():
    return torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"


# ---------------------------------------------------------------- logging
RESULT_FIELDS = [
    "method", "seed", "device",
    "test_acc", "retain_acc", "forget_acc",
    "canary_loss", "canary_memorized",
    "train_time_s", "request_time_s", "peak_mem_mb", "gpu_hours", "usd",
]


def log_result(row, path="results/results.csv"):
    """Append one row to the shared results CSV (all methods write here)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
        