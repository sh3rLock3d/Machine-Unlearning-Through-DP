"""SISA (Bourtoule et al., 2021): exact unlearning via Sharded, Isolated, Sliced, Aggregated training.

1. Split the training set into 5 disjoint shards; train one model per shard (isolated).
2. Each shard is split into slices; the shard model is trained incrementally, slice by slice,
   saving a checkpoint after each slice.
3. Prediction = average of the shard models' softmax outputs (aggregated).
4. Deletion: for each shard containing deleted points, reload the checkpoint from just before the
   first affected slice and retrain from there without the deleted points. Other shards are untouched.

Two rows are logged per run:
  sisa_full               -- SISA trained on everything (before the deletion request)
  sisa_unlearn_<exclude>  -- after handling the deletion request

Usage:
    python sisa.py --exclude forget --seed 0
    python sisa.py --exclude canary --seed 0
"""
import argparse
import os

import torch
import torch.nn as nn

from common import (DEVICE, CostTracker, SmallCNN, accuracy, apply_canary, device_name,
                    load_mnist, log_result, make_splits, per_sample_loss, set_seed, train)

p = argparse.ArgumentParser()
p.add_argument("--exclude", choices=["forget", "canary"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--shards", type=int, default=5)
p.add_argument("--slices", type=int, default=4)
p.add_argument("--epochs-per-slice", type=int, default=2)
args = p.parse_args()


class Ensemble(nn.Module):
    """Averages shard models' softmax; returns log-probs so cross_entropy gives the ensemble's loss."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        probs = torch.stack([m(x).softmax(1) for m in self.models]).mean(0)
        return probs.clamp_min(1e-12).log()


def ckpt_path(shard, slc):
    return os.path.join(ckpt_dir, f"shard{shard}_slice{slc}.pt")


def train_shard(shard, start_slice):
    """Train shard model from the checkpoint before `start_slice` through the last slice."""
    model = SmallCNN().to(DEVICE)
    if start_slice > 0:
        model.load_state_dict(torch.load(ckpt_path(shard, start_slice - 1)))
    for k in range(start_slice, args.slices):
        seen = torch.cat(slices[shard][:k + 1])  # incremental: slices 0..k
        train(model, x_train, y_train, seen, epochs=args.epochs_per_slice)
        torch.save(model.state_dict(), ckpt_path(shard, k))
    return model


def evaluate(model, method, train_time, request_time, cost):
    canary_loss, canary_correct = per_sample_loss(model, x_train[canary:canary + 1], y_train[canary:canary + 1])
    storage_mb = sum(os.path.getsize(os.path.join(ckpt_dir, f)) for f in os.listdir(ckpt_dir)) / 2**20
    row = {
        "method": method,
        "seed": args.seed,
        "device": device_name(),
        "test_acc": accuracy(model, x_test, y_test),
        "retain_acc": accuracy(model, x_train[retain], y_train[retain]),
        "forget_acc": accuracy(model, x_train[forget], y_train[forget]),
        "canary_loss": canary_loss.item(),
        "canary_memorized": int(canary_correct.item()),
        "train_time_s": train_time,
        "request_time_s": request_time,
        "peak_mem_mb": cost.peak_mem_mb,
        "gpu_hours": cost.gpu_hours,
        "usd": cost.usd,
        "storage_mb": storage_mb,
    }
    log_result(row)
    print()
    for k, v in row.items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")


# ---------------------------------------------------------------- setup
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

ckpt_dir = f"checkpoints/sisa_{args.exclude}_seed{args.seed}"
os.makedirs(ckpt_dir, exist_ok=True)

# random shard/slice assignment: slices[shard][slice] = tensor of training indices
perm = all_idx[torch.randperm(n, device=DEVICE)]
slices = [list(shard.chunk(args.slices)) for shard in perm.chunk(args.shards)]

# ---------------------------------------------------------------- 1) initial SISA training
with CostTracker() as cost_train:
    models = [train_shard(s, 0) for s in range(args.shards)]
evaluate(Ensemble(models), "sisa_full", cost_train.seconds, "", cost_train)

# ---------------------------------------------------------------- 2) deletion request
to_delete = forget if args.exclude == "forget" else torch.tensor([canary], device=DEVICE)

with CostTracker() as cost_req:
    affected = []
    for s in range(args.shards):
        hit = [torch.isin(slc, to_delete).any().item() for slc in slices[s]]
        if not any(hit):
            continue  # shard never saw deleted data -> untouched
        first = hit.index(True)
        slices[s] = [slc[~torch.isin(slc, to_delete)] for slc in slices[s]]
        models[s] = train_shard(s, first)
        affected.append((s, first))

print(f"\nAffected (shard, first slice retrained from): {affected}")
evaluate(Ensemble(models), f"sisa_unlearn_{args.exclude}", cost_train.seconds, cost_req.seconds, cost_req)