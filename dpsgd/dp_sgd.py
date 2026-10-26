"""DP-SGD (Abadi et al., 2016) with Opacus: bound every record's influence during training.

DP does nothing when a deletion request arrives -- the model trained on the full data simply stays
deployed, so its request cost is 0. Two variants:
  --exclude none     train on everything (the deployed DP model)     -> accuracy / MIA / cost
  --exclude canary   train without the canary                        -> other side of the epsilon audit

Usage:
    python dp_sgd.py --epsilon 8 --exclude none --seed 0
    for e in 1 3 8; do python dp_sgd.py --epsilon $e --exclude none --seed 0; done
"""
import argparse
import os
import warnings

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

from common import (DEVICE, CostTracker, SmallCNN, accuracy, apply_canary, device_name,
                    load_mnist, log_result, make_splits, per_sample_loss, set_seed)

warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Full backward hook.*")  # harmless Opacus warning

p = argparse.ArgumentParser()
p.add_argument("--epsilon", type=float, required=True)
p.add_argument("--exclude", choices=["none", "canary"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--delta", type=float, default=1e-5)  # < 1/n for n = 60,000
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch-size", type=int, default=256)  # expected batch size (Poisson sampling)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max-grad-norm", type=float, default=1.0)  # per-sample clipping bound C
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

keep = torch.ones(n, dtype=torch.bool, device=DEVICE)
if args.exclude == "canary":
    keep[canary] = False
train_idx = all_idx[keep]

# Opacus needs a DataLoader for Poisson sampling; keep it on CPU and move each batch to the GPU.
loader = DataLoader(TensorDataset(x_train[train_idx].cpu(), y_train[train_idx].cpu()),
                    batch_size=args.batch_size, shuffle=True)

model = SmallCNN().to(DEVICE)
assert not ModuleValidator.validate(model, strict=False), "model has layers Opacus can't handle"
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private_with_epsilon(
    module=model, optimizer=optimizer, data_loader=loader,
    target_epsilon=args.epsilon, target_delta=args.delta,
    epochs=args.epochs, max_grad_norm=args.max_grad_norm,
)

# ---- DP training (this is DP's whole cost; nothing happens per deletion request)
with CostTracker() as cost:
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            if len(xb) == 0:  # Poisson sampling can produce empty batches
                continue
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()

spent_eps = privacy_engine.get_epsilon(args.delta)
net = model._module  # unwrap Opacus' GradSampleModule for evaluation / saving

# ---- evaluate (same metrics as every other method)
tag = f"dp_eps{args.epsilon:g}_{'full' if args.exclude == 'none' else 'nocanary'}"
canary_loss, canary_correct = per_sample_loss(net, x_train[canary:canary + 1], y_train[canary:canary + 1])
row = {
    "method": tag,
    "seed": args.seed,
    "device": device_name(),
    "test_acc": accuracy(net, x_test, y_test),
    "retain_acc": accuracy(net, x_train[retain], y_train[retain]),
    "forget_acc": accuracy(net, x_train[forget], y_train[forget]),
    "canary_loss": canary_loss.item(),
    "canary_memorized": int(canary_correct.item()),
    "train_time_s": cost.seconds,
    "request_time_s": 0.0 if args.exclude == "none" else "",  # DP: no action on a deletion request
    "peak_mem_mb": cost.peak_mem_mb,
    "gpu_hours": cost.gpu_hours,
    "usd": cost.usd,
}
log_result(row)

os.makedirs("checkpoints", exist_ok=True)
torch.save(net.state_dict(), f"checkpoints/{tag}_seed{args.seed}.pt")

for k, v in row.items():
    print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
print(f"{'epsilon spent':>16}: {spent_eps:.3f} (target {args.epsilon:g}, delta {args.delta:g})")
print(f"{'noise multiplier':>16}: {optimizer.noise_multiplier:.3f}")