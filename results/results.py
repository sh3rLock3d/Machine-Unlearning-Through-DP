import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
                     "axes.linewidth": 0.6})

methods = ["Original", "Retrain", "SISA", "NegGrad+", r"DP $\varepsilon$=8", r"DP $\varepsilon$=3", r"DP $\varepsilon$=1"]
test   = [98.88, 98.90, 98.65, 99.10, 96.91, 96.47, 95.07]
forget = [99.67, 99.17, 98.67, 99.83, 97.17, 96.33, 95.33]
train  = [5.22, 4.78, 5.33, 5.22, 119.47, 114.08, 113.09]   # NegGrad+ starts from the original model
req    = [None, 4.78, 5.43, 1.94, 0.0, 0.0, 0.0]

y = np.arange(len(methods))[::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"wspace": 0.55})

# (a) accuracy dot plot
for i in range(len(methods)):
    a.plot([test[i], forget[i]], [y[i], y[i]], color=GRID, lw=2, zorder=1)
a.scatter(test, y, s=30, color=BLUE, marker="o", label="Test", zorder=3, edgecolor="white", linewidth=0.8)
a.scatter(forget, y, s=34, color=ORANGE, marker="D", label="Forget set", zorder=3, edgecolor="white", linewidth=0.8)
a.axvline(forget[1], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
a.text(forget[1] - 0.08, y[0] + 0.75, "retrain\nforget acc.", ha="right", va="bottom", fontsize=6.5, color=INK2)
a.set_yticks(y, methods); a.set_xlim(94.5, 100.3); a.set_ylim(-0.6, len(methods) + 0.4)
a.set_xlabel("Accuracy (%)"); a.set_title("(a) Utility and forget-set accuracy", fontsize=8.5, color=INK, loc="left")
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.3)

# (b) cost, log scale
h = 0.36
b.barh(y + h / 2 + 0.01, train, height=h, color=BLUE, label="Initial training", edgecolor="white", linewidth=1)
rv = [r if r else np.nan for r in req]
b.barh(y - h / 2 - 0.01, rv, height=h, color=ORANGE, label="Per deletion request", edgecolor="white", linewidth=1)
for i, r in enumerate(req):
    yy = y[i] - h / 2 - 0.01
    if r is None:
        b.text(1.15, yy, "n/a", va="center", fontsize=6.5, color=INK2)
    elif r == 0:
        b.text(1.15, yy, "0 (no action)", va="center", fontsize=6.5, color=INK2)
    else:
        b.text(r * 1.12, yy, f"{r:.1f}s", va="center", fontsize=6.5, color=INK)
for i in (4, 5, 6):
    b.text(train[i] * 1.1, y[i] + h / 2, f"{train[i]:.0f}s", va="center", fontsize=6.5, color=INK)
b.set_xscale("log"); b.set_xlim(1, 400)
b.set_yticks(y, methods); b.set_ylim(-0.6, len(methods) + 0.4)
b.set_xlabel("Wall-clock time, CPU (s, log scale)")
b.set_title("(b) Computational cost", fontsize=8.5, color=INK, loc="left")
b.grid(axis="x", color=GRID, lw=0.6, which="major"); b.set_axisbelow(True)
b.legend(loc="upper right", frameon=False, fontsize=7, handletextpad=0.3)

for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

fig.savefig("results_fig.pdf", bbox_inches="tight")
fig.savefig("results_fig.png", dpi=220, bbox_inches="tight")