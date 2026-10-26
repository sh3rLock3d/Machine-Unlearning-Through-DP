"""NegGrad+ (Kurmanji et al., 2023): approximate unlearning starting from the ORIGINAL model.

Each step takes one batch of retained data and one batch of forget data and minimizes
    alpha * loss(retain) - (1 - alpha) * loss(forget)
i.e. gradient DESCENT on retained data (keep utility) and gradient ASCENT on forget data (forget it).
alpha = 0 gives plain NegGrad (pure gradient ascent on the forget data).

Gradient ascent is unbounded: without a limit the forget loss goes to infinity and the model collapses.
Each forget sample's loss is therefore capped at ln(10) ~ 2.30, the loss of a uniform (chance-level)
prediction; once a sample reaches it, ascent on that sample stops.

Requires checkpoints/original_seed{seed}.pt (run train_original.py first).

Usage:
    python neggrad.py --exclude forget --seed 0
    for s in $(seq 0 199); do python neggrad.py --exclude canary --seed $s; done
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F

from common import (DEVICE, CostTracker, SmallCNN, accuracy, apply_canary, device_name,
                    load_mnist, log_result, make_splits, per_sample_loss, set_seed)

p = argparse.ArgumentParser()
p.add_argument("--exclude", choices=["forget", "canary"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--alpha", type=float, default=0.99)  # weight on retain loss; 0 = plain NegGrad
p.add_argument("--epochs", type=int, default=1)      # passes over the retained data
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--forget-loss-cap", type=float, default=math.log(10))  # stop ascent at chance level
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

# ---- what to forget / keep for this request (matches retrain.py's two variants)
delete = forget if args.exclude == "forget" else torch.tensor([canary], device=DEVICE)
keep = torch.ones(n, dtype=torch.bool, device=DEVICE)
keep[delete] = False
keep_idx = all_idx[keep]

# ---- start from the original model
model = SmallCNN().to(DEVICE)
model.load_state_dict(torch.load(f"../original/checkpoints/original_seed{args.seed}.pt", map_location=DEVICE))

# ---- NegGrad+ (this is the cost of handling the deletion request)
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
fb = min(args.batch_size, len(delete))  # forget batch size (1 for the canary)
with CostTracker() as cost:
    model.train()
    for _ in range(args.epochs):
        perm_keep = keep_idx[torch.randperm(len(keep_idx), device=DEVICE)]
        for i in range(0, len(perm_keep), args.batch_size):
            rb = perm_keep[i:i + args.batch_size]
            fbatch = delete[torch.randint(len(delete), (fb,), device=DEVICE)]
            forget_loss = F.cross_entropy(model(x_train[fbatch]), y_train[fbatch], reduction="none")
            loss = (args.alpha * F.cross_entropy(model(x_train[rb]), y_train[rb])
                    - (1 - args.alpha) * forget_loss.clamp(max=args.forget_loss_cap).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

# ---- evaluate (same metrics as every other method)
tag = f"neggrad_a{args.alpha:g}_{args.exclude}"
canary_loss, canary_correct = per_sample_loss(model, x_train[canary:canary + 1], y_train[canary:canary + 1])
row = {
    "method": tag,
    "seed": args.seed,
    "device": device_name(),
    "test_acc": accuracy(model, x_test, y_test),
    "retain_acc": accuracy(model, x_train[retain], y_train[retain]),
    "forget_acc": accuracy(model, x_train[forget], y_train[forget]),
    "canary_loss": canary_loss.item(),
    "canary_memorized": int(canary_correct.item()),
    "train_time_s": "",  # initial training = the original model's cost
    "request_time_s": cost.seconds,
    "peak_mem_mb": cost.peak_mem_mb,
    "gpu_hours": cost.gpu_hours,
    "usd": cost.usd,
}
log_result(row)

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/{tag}_seed{args.seed}.pt")

for k, v in row.items():
    print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")