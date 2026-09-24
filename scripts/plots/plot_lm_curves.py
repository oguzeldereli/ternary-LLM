"""LM 10M screens: raw train-loss curves (light) with a smoothed line on top, linear axes,
full view and the last-third zoom; final validation loss in the legend.

  python -m scripts.plots.plot_lm_curves
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color
    ("lm_lowrank256_xb2", "rank-256 momentum (fixed 0.97) + look-ahead x2", "#e53935"),
    ("la_xb2_rc", "look-ahead x2 + row/col scales", "#2a78d6"),
    ("la_xb2", "look-ahead x2", "#111111"),
    ("lm_lowrank256_adapt_xb2", "adaptive momentum + look-ahead x2 (stopped @200)", "#00897b"),
    ("la_fast_fp32tail", "same-batch look-ahead", "#c2185b"),
    ("lm_plain", "plain flips", "#1baf7a"),
    ("lm_frozen", "frozen random ternary", "#8e8c85"),
    ("lm_lowrank256", "rank-256 momentum, no look-ahead", "#7b3fb8"),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r}
    v = [r["val_loss"] for r in rows if r.get("final") and "val_loss" in r]
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    return s, L, (v[-1] if v else None)


def smooth(y, k=15):
    c = np.cumsum(np.insert(y, 0, 0.0))
    w = [min(k, i // 3 + 1) for i in range(len(y))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(y))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/lm_curves.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor=SURFACE)
for d, lab, col in RUNS:
    s, L, v = load(d)
    lab = f"{lab}  (val {v:.3f})" if v else lab
    for ax in (ax1, ax2):
        ax.plot(s, L, color=col, lw=0.7, alpha=0.25)
        ax.plot(s, smooth(L), color=col, lw=2.4, label=lab)
ax1.set_ylim(4.6, 10.8); ax1.set_xlim(0, 320)
ax2.set_xlim(200, 318); ax2.set_ylim(4.7, 6.6)
for ax, tt in zip((ax1, ax2), ("Train loss, 10M tokens (raw faint, 15-step mean bold)",
                               "Zoom: steps 200-316")):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("step (32,768 tokens each)", color=INK2); ax.set_ylabel("loss", color=INK2)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
ax1.legend(fontsize=9.5, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
