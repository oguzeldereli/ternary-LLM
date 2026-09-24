"""Flip-rate screens (10M tokens): train loss, difference vs cross-batch look-ahead at
rate 0.02, and flips per step.

  python -m scripts.plots.plot_rates
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BASE = "la_xb2"
RUNS = [  # dir, label, color, style
    ("p2_baseline", "master weights", "#eb6834", "-"),
    ("la_xb2", "cross-batch look-ahead x2, rate 0.02", "#111111", "-"),
    ("la_xb2_r0.125", "cross-batch look-ahead x2, rate 0.125", "#c2185b", "-"),
    ("la_xb2_r0.004", "cross-batch look-ahead x2, rate 0.004", "#1baf7a", "-"),
    ("la_fast_fp32tail", "same-batch look-ahead, rate 0.02", "#2a78d6", "--"),
    ("la_fast_fp32tail_r04", "same-batch look-ahead, rate 0.04", "#7b3fb8", "--"),
]
MAX_STEP = 320


def load(d):
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    tr = {}
    for l in open(f):
        r = json.loads(l)
        if "loss" in r and "tokens" in r and not r.get("probe") and r["step"] <= MAX_STEP:
            tr[r["step"]] = r
    return tr or None


def smooth(L, k=20):
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/rates.png")
a = ap.parse_args()
D = {d: load(d) for d, *_ in RUNS}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)
base = D[BASE]
for d, lab, c, ls in RUNS:
    tr = D[d]
    if tr is None: continue
    s = sorted(tr); t = np.array([tr[i]["tokens"] for i in s], float)
    ax1.plot(t, smooth(np.array([tr[i]["loss"] for i in s])), color=c, ls=ls, lw=2.4, label=lab)
    if d not in (BASE, "p2_baseline"):
        s2 = [i for i in s if i in base]
        ax2.plot([tr[i]["tokens"] for i in s2],
                 smooth(np.array([tr[i]["loss"] - base[i]["loss"] for i in s2])),
                 color=c, ls=ls, lw=2.4, label=lab)
    if "flip_frac" in tr[s[-1]]:
        ax3.plot(t, smooth(np.array([tr[i].get("flip_frac", 0) * 100 for i in s]), 10),
                 color=c, ls=ls, lw=2.2, label=lab)
ax2.axhline(0, color=INK2, lw=1)
titles = ("Train loss (first 10M tokens)", "Train loss minus cross-batch x2 @ 0.02 (same batches)",
          "Flips applied per step (% of weights)")
for ax, tt in zip((ax1, ax2, ax3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_xscale("log"); ax.set_xlim(3e4, 1.1e7)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=8.5, frameon=False)
ax1.set_yscale("log"); ax1.set_ylim(4.9, 11)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
