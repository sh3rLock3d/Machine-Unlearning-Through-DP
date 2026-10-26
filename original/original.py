"""Train the ORIGINAL model: full training set (forget set + canary included).

Usage:
    python train_original.py --seed 0
    for s in 0 1 2 3 4; do python train_original.py --seed $s; done
"""
import argparse
import os

import torch

from retrain.common import (DEVICE, CostTracker, SmallCNN, accuracy, apply_canary, device_name,
                    load_mnist, log_result, make_splits, per_sample_loss, set_seed, train)

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=5)
args = p.parse_args()

# The splits use their own fixed seed (0), so every run and method shares the same forget set and canary.
(x_train, y_train), (x_test, y_test) = load_mnist()
splits = make_splits(y_train)
apply_canary(y_train, splits)

set_seed(args.seed)
n = len(x_train)
canary = splits["canary_idx"]
forget = torch.tensor(splits["forget_idx"], device=DEVICE)
all_idx = torch.arange(n, device=DEVICE)
retain_mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
retain_mask[forget] = False
retain_mask[canary] = False
retain = all_idx[retain_mask]

# ---- train on everything
model = SmallCNN().to(DEVICE)
with CostTracker() as cost:
    train(model, x_train, y_train, all_idx, epochs=args.epochs)

# ---- evaluate
canary_loss, canary_correct = per_sample_loss(model, x_train[canary:canary + 1], y_train[canary:canary + 1])
row = {
    "method": "original",
    "seed": args.seed,
    "device": device_name(),
    "test_acc": accuracy(model, x_test, y_test),
    "retain_acc": accuracy(model, x_train[retain], y_train[retain]),
    "forget_acc": accuracy(model, x_train[forget], y_train[forget]),
    "canary_loss": canary_loss.item(),
    "canary_memorized": int(canary_correct.item()),  # 1 = model predicts the flipped label
    "train_time_s": cost.seconds,
    "request_time_s": "",  # no deletion request for the original model
    "peak_mem_mb": cost.peak_mem_mb,
    "gpu_hours": cost.gpu_hours,
    "usd": cost.usd,
}
log_result(row)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/original_seed{args.seed}.pt")

for k, v in row.items():
    print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")