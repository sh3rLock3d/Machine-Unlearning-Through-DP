"""Membership inference on the forget set, for every saved model: the threshold attack of
Yeom et al. (CSF 2018). A sample is predicted "member" if the model's loss on it is below the
model's average training loss L_S. We report the membership advantage = TPR - FPR, where
TPR = fraction of the 600 forget samples predicted member, FPR = fraction of 600 test samples
predicted member. 0 = no leakage (forget set looks like unseen data).
Compare each method against the RETRAIN model's advantage, which is the target.

Place this file in ~/test/results/ and run it from there:
    cd ~/test/results && python evaluate_mia.py
It reads checkpoints from the sibling method folders and writes mia.csv here.
"""
import csv
import glob
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "../original")  # reuse common.py from the original folder
from common import DEVICE, SmallCNN, apply_canary, load_mnist, make_splits, per_sample_loss  # noqa: E402

N_SISA_SHARDS, N_SISA_SLICES = 5, 4  # must match sisa.py defaults


class Ensemble(nn.Module):  # same aggregation as sisa.py
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        probs = torch.stack([m(x).softmax(1) for m in self.models]).mean(0)
        return probs.clamp_min(1e-12).log()


def load_cnn(path):
    m = SmallCNN().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m


def membership_advantage(member_loss, nonmember_loss, threshold):
    """Yeom et al.: predict member if loss < threshold; advantage = TPR - FPR."""
    tpr = (member_loss < threshold).float().mean().item()
    fpr = (nonmember_loss < threshold).float().mean().item()
    return tpr - fpr, tpr, fpr


# ---------------------------------------------------------------- data (same splits as training)
(x_train, y_train), (x_test, y_test) = load_mnist(root="../original/data")
splits = make_splits(y_train, path="../original/splits.json")
apply_canary(y_train, splits)
forget = torch.tensor(splits["forget_idx"], device=DEVICE)
g = torch.Generator().manual_seed(0)
test_sub = torch.randperm(len(x_test), generator=g)[:len(forget)].to(DEVICE)  # fixed 600 test samples
keep = torch.ones(len(x_train), dtype=torch.bool)
keep[forget.cpu()] = False
keep[splits["canary_idx"]] = False
retain_sub = keep.nonzero().squeeze(1)[torch.randperm(int(keep.sum()), generator=g)[:10000]].to(DEVICE)

# ---------------------------------------------------------------- find models trained for the forget-set request
models = []  # (method, seed, loader)
pattern = re.compile(r"(original|retrain_forget|neggrad_a[\d.]+_forget|dp_eps[\d.]+_full)_seed(\d+)\.pt$")
for path in sorted(glob.glob("../*/checkpoints/*.pt")):
    m = pattern.search(os.path.basename(path))
    if m:
        models.append((m.group(1), int(m.group(2)), lambda p=path: load_cnn(p)))
for d in sorted(glob.glob("../*/checkpoints/sisa_forget_seed*")):
    seed = int(re.search(r"seed(\d+)$", d).group(1))
    files = [os.path.join(d, f"shard{s}_slice{N_SISA_SLICES - 1}.pt") for s in range(N_SISA_SHARDS)]
    if all(os.path.exists(f) for f in files):  # final (post-deletion) shard models
        models.append(("sisa_unlearn_forget", seed, lambda fs=files: Ensemble([load_cnn(f) for f in fs])))

if not models:
    sys.exit("No checkpoints found. Run this from ~/test/results/.")

# ---------------------------------------------------------------- attack each model
rows = []
for method, seed, load in models:
    model = load()
    f_loss, _ = per_sample_loss(model, x_train[forget], y_train[forget])
    t_loss, _ = per_sample_loss(model, x_test[test_sub], y_test[test_sub])
    r_loss, _ = per_sample_loss(model, x_train[retain_sub], y_train[retain_sub])
    threshold = r_loss.mean().item()  # L_S: average loss on data every model trained on
    adv, tpr, fpr = membership_advantage(f_loss, t_loss, threshold)
    rows.append({
        "method": method,
        "seed": seed,
        "mia_advantage": adv,
        "tpr": tpr,
        "fpr": fpr,
        "threshold": threshold,
        "forget_loss": f_loss.mean().item(),
        "test_loss": t_loss.mean().item(),
    })

with open("mia.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"{'method':<24}{'seed':>5}{'advantage':>11}{'TPR':>8}{'FPR':>8}")
for r in rows:
    print(f"{r['method']:<24}{r['seed']:>5}{r['mia_advantage']:>11.4f}{r['tpr']:>8.4f}{r['fpr']:>8.4f}")
print("\nSaved mia.csv. Target = the retrain_forget advantage (not exactly 0).")