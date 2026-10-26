"""RETRAIN (gold standard): train from scratch without the data to be deleted.

Two variants, one per experiment:
  --exclude forget   drop the 600-sample forget set (canary kept)  -> reference for accuracy / MIA
  --exclude canary   drop only the canary (forget set kept)        -> reference for empirical epsilon

Usage:
    python retrain.py --exclude forget --seed 0
    for s in $(seq 0 199); do python retrain.py --exclude canary --seed $s; done
"""
import argparse
import os

import torch

from common import (DEVICE, CostTracker, SmallCNN, accuracy, apply_canary, device_name,
                    load_mnist, log_result, make_splits, per_sample_loss, set_seed, train)

p = argparse.ArgumentParser()
p.add_argument("--exclude", choices=["forget", "canary"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=5)
args = p.parse_args()

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

# ---- training set = everything except the deleted data
keep = torch.ones(n, dtype=torch.bool, device=DEVICE)
if args.exclude == "forget":
    keep[forget] = False
else:
    keep[canary] = False
train_idx = all_idx[keep]

# ---- retrain from scratch (this IS the cost of handling the deletion request)
model = SmallCNN().to(DEVICE)
with CostTracker() as cost:
    train(model, x_train, y_train, train_idx, epochs=args.epochs)

# ---- evaluate (same metrics as the original model)
canary_loss, canary_correct = per_sample_loss(model, x_train[canary:canary + 1], y_train[canary:canary + 1])
row = {
    "method": f"retrain_{args.exclude}",
    "seed": args.seed,
    "device": device_name(),
    "test_acc": accuracy(model, x_test, y_test),
    "retain_acc": accuracy(model, x_train[retain], y_train[retain]),
    "forget_acc": accuracy(model, x_train[forget], y_train[forget]),
    "canary_loss": canary_loss.item(),
    "canary_memorized": int(canary_correct.item()),
    "train_time_s": cost.seconds,
    "request_time_s": cost.seconds,  # a deletion request costs one full retrain
    "peak_mem_mb": cost.peak_mem_mb,
    "gpu_hours": cost.gpu_hours,
    "usd": cost.usd,
}
log_result(row)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/retrain_{args.exclude}_seed{args.seed}.pt")

for k, v in row.items():
    print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")