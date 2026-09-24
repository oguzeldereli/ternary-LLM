"""Movement geometry along a full MLP run (scripts/mlp/geometry.py output):
alignment of each 200-step window's move, and of the whole move since step 0, with the
summed gradient; trits changed per window.

  python -m scripts.plots.plot_geometry
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MASTER, FLIP = "#eb6834", "#1baf7a"

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/geometry.png")
a = ap.parse_args()
R = json.load(open("checkpoints/mlp/geometry_full.json"))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)
ex = lambda rows: np.array([r["step"] for r in rows]) * 4096
m, f, x2 = R["master"], R.get("flip", []), R.get("xb2", [])
XB = "#111111"
ax1.plot(ex(m), [r["win_latent"] for r in m], color=MASTER, lw=1.6, ls="--", label="master: float weights")
ax1.plot(ex(m), [r["win_trit"] for r in m], color=MASTER, lw=2.4, label="master: trits")
if f: ax1.plot(ex(f), [r["win_trit"] for r in f], color=FLIP, lw=2.4, label="stateless flips: trits")
if x2: ax1.plot(ex(x2), [r["win_trit"] for r in x2], color=XB, lw=2.4, label="flips + cross-batch look-ahead x2: trits")
ax2.plot(ex(m), [r["cum_latent"] for r in m], color=MASTER, lw=1.6, ls="--", label="master: float weights")
ax2.plot(ex(m), [r["cum_trit"] for r in m], color=MASTER, lw=2.4, label="master: trits")
if f: ax2.plot(ex(f), [r["cum_trit"] for r in f], color=FLIP, lw=2.4, label="stateless flips: trits")
if x2: ax2.plot(ex(x2), [r["cum_trit"] for r in x2], color=XB, lw=2.4, label="flips + cross-batch look-ahead x2: trits")
ax3.plot(ex(m), [r["win_changed_%"] for r in m], color=MASTER, lw=2.4, label="master")
if f: ax3.plot(ex(f), [r["win_changed_%"] for r in f], color=FLIP, lw=2.4, label="stateless flips")
if x2: ax3.plot(ex(x2), [r["win_changed_%"] for r in x2], color=XB, lw=2.4, label="flips + cross-batch look-ahead x2")
titles = ("Each 200-step window: move vs -summed gradient (cosine)",
          "Everything since step 0: move vs -summed gradient (cosine)",
          "Trits changed per 200-step window (%)")
for ax, tt in zip((ax1, ax2, ax3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_xscale("log"); ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("training examples", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
ax1.axhline(0, color=INK2, lw=1); ax2.axhline(0, color=INK2, lw=1)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
