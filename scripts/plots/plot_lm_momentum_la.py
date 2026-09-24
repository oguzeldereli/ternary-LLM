"""Momentum + look-ahead on the LM vs look-ahead alone (la_xb2), per step: train-loss
difference, cos(g, M), flips applied per step.

  python -m scripts.plots.plot_lm_momentum_la
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BASE = "la_xb2"
RUNS = [  # dir, label, color
    ("lm_lowrank256_xb2", "rank-256 momentum, fixed beta 0.97 + look-ahead x2", "#e53935"),
    ("lm_lowrank256_adapt_xb2", "rank-256 momentum, decay 0.97*cos + look-ahead x2 (stopped)", "#00897b"),
    ("lm_plain", "plain flips, no look-ahead", "#1baf7a"),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    return {r["step"]: r for r in rows if "flip_frac" in r}


def smooth(y, k=20):
    c = np.cumsum(np.insert(y, 0, 0.0))
    w = [min(k, i // 3 + 1) for i in range(len(y))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(y))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/lm_momentum_la.png")
a = ap.parse_args()
B = load(BASE)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)
bs = sorted(B)
ax3.plot(bs, smooth(np.array([B[s]["flip_frac"] * 100 for s in bs])), color="#111111", lw=2.4,
         label="look-ahead x2 alone (la_xb2)")
for d, lab, col in RUNS:
    R = load(d); st = [s for s in sorted(R) if s in B]
    diff = np.array([R[s]["loss"] - B[s]["loss"] for s in st])
    ax1.plot(st, smooth(diff), color=col, lw=2.4, label=lab)
    cs = [(s, R[s]["lr_cos"]) for s in st if "lr_cos" in R[s]]
    if cs: ax2.plot([s for s, _ in cs], smooth(np.array([c for _, c in cs]), 5), color=col, lw=2.2, label=lab)
    ax3.plot(st, smooth(np.array([R[s]["flip_frac"] * 100 for s in st])), color=col, lw=2.2, label=lab)
R = load("lm_lowrank256_adapt")
cs = [(s, R[s]["lr_cos"]) for s in sorted(R) if "lr_cos" in R[s]]
ax2.plot([s for s, _ in cs], smooth(np.array([c for _, c in cs]), 5), color="#e0a100", lw=2.0,
         label="rank-256 momentum, decay 0.97*cos, no look-ahead")
for ax in (ax1, ax2): ax.axhline(0, color=INK2, lw=1)
ax1.set_ylim(-0.3, 0.8); ax2.set_ylim(-1, 1)
titles = ("Train loss minus look-ahead x2 alone (same batches; <0 = better)",
          "cos(gradient, momentum), mean over 84 layers",
          "Trits changed per step (%)")
for ax, tt in zip((ax1, ax2, ax3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("step (32,768 tokens each)", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
