"""Look-ahead full run against the reference runs; safe to re-run while it trains.

  python3 plot_live.py --out docs/lookahead_live.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width
    ("p2_baseline",   "master weights (ceiling)",           "#eb6834", 2.2),
    ("armA_cos_la1",  "LOOK-AHEAD (running)",               "#c2185b", 3.0),
    ("armA_cosine",   "flips, cosine rate (reference)",     "#1baf7a", 2.2),
    ("armA_cos_r005", "flips, cosine from 4x lower rate",   "#2a78d6", 1.4),
    ("armA_cos_600M", "flips, cosine rate, 600M schedule",  "#8e8c85", 1.4),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = [r for r in rows if "loss" in r and "tokens" in r]
    t = np.array([r["tokens"] for r in tr], float); L = np.array([r["loss"] for r in tr])
    tok_per_step = t[1] - t[0]
    v = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    vt = np.array([(s + 1) * tok_per_step for s, _ in v], float)
    return t, L, vt, np.array([np.exp(x) for _, x in v])


def smooth(L, k=100):
    # trailing mean over min(k, ~20% of steps so far): a fixed window drags the
    # starting loss into the first k points and flattens the fast early drop
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/lookahead_live.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor=SURFACE)
for d, lab, c, lw in RUNS:
    t, L, vt, vp = load(d)
    ax1.plot(t, smooth(L), color=c, lw=lw, label=lab, zorder=6 if "LOOK" in lab else 4)
    if len(vp):
        ax2.plot(vt, vp, "o-", color=c, lw=lw, ms=5, label=lab, zorder=6 if "LOOK" in lab else 4)
        if "LOOK" in lab:
            ax2.annotate(f"{vp[-1]:.1f}", (vt[-1], vp[-1]), textcoords="offset points",
                         xytext=(8, 4), color=c, fontsize=11, weight="bold")
for ax, yl, title in ((ax1, "train loss (100-step avg)", "Train loss, log-log"),
                      (ax2, "val perplexity", "Validation perplexity (every 1000 steps)")):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.set_ylabel(yl, color=INK2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.legend(fontsize=9, frameon=False)
ax1.set_xlim(3e4, 7e8); ax1.set_ylim(2.5, 11)
ax2.set_xlim(2.5e7, 7e8)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=130, facecolor=SURFACE)
print("wrote", a.out)
