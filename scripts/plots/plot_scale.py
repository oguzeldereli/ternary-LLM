"""Rank and model size for the current recipe (sharp base + beta 1 + dry friction 1/33 + spend 3). Left: the 110M rank
sweep, validation loss 100M-300M; middle: 340M (d1024_l24) runs against 110M, with master at both sizes; right: final
loss against rank at both sizes (the gap between the two lines is what the bigger model buys; the slope is the rank
penalty). Unfinished runs are dashed and labelled with their step.

  python -m scripts.plots.plot_scale      -> docs/figures/scale.png
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.plots.plot_best import load, SURFACE, INK, INK2, GRID

RANKS = [(32, "rk32_dryspend_s0"), (64, "rk64_dryspend_s0"), (128, "rk128_dryspend_s0"), (256, "gvsharp_dryspend_s0"),
         (512, "gvsharp_dryspend_r512_s0"), (1024, "gvsharp_dryspend_r1024_s0")]
EXTRA = [("rk128_int8_dryspend_s0", "rank 128, int8 momentum"), ("rk64_refresh_dryspend_s0", "rank 64 + subspace refresh")]
BIG = [(64, "big_rk64_dryspend_s0"), (128, "big_rk128_dryspend_s0"), (512, "big_dryspend_r512_s0"), (1024, "big_dryspend_r1024_s0")]
SHADE = {32: 0.25, 64: 0.38, 128: 0.52, 256: 0.66, 512: 0.82, 1024: 1.0}
C110, C340 = "#b3261e", "#1565c0"
MASTER = "#eb6834"


def end(r):
    t, v, done, fin = r
    return fin if fin is not None else v[-1]


fig, axs = plt.subplots(1, 3, figsize=(24, 8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
for ax in axs:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")

a = axs[0]
ENDS = []
def curve(ax, r, c, lw, ls, lab, alpha=1.0):
    t, v, done, fin = r
    m = t >= 9.5e7
    ax.plot(t[m], v[m], ls if done else "--", color=c, lw=lw, alpha=alpha, label=lab)
    e = end(r); y = e
    while any(abs(y - u) < 0.014 for u in ENDS): y += 0.014
    ENDS.append(y)
    ax.annotate(f"{e:.3f}" + ("" if done else f" @{t[-1] / 1e6:.0f}M"), (t[m][-1], y), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=9, color=c, alpha=max(alpha, 0.6))
m110 = load("master_tracked")
curve(a, m110, MASTER, 2.2, "-", f"master weights  [{end(m110):.3f}]")
for k, run in RANKS:
    r = load(run)
    if r: curve(a, r, C110, 2.0, "-", f"rank {k}  [{end(r):.3f}]", SHADE[k])
for run, lab in EXTRA:
    r = load(run)
    if r: curve(a, r, "#546e7a", 1.4, ":", f"{lab}  [{end(r):.3f}]")
a.set_xlim(9.5e7, 3.4e8); a.set_ylim(2.72, 3.3); a.xaxis.set_major_formatter(fmt)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss (nats per token)", color=INK2)
a.set_title("110M: momentum rank (dry friction + spend)", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)

a = axs[1]
ENDS = []
m340 = load("big_master")
if m340: curve(a, m340, MASTER, 2.4, "-", f"master weights, 340M  [{end(m340):.3f}" + ("]" if m340[2] else ", running]"))
curve(a, m110, MASTER, 1.4, "-", f"master weights, 110M  [{end(m110):.3f}]", 0.45)
for k, run in BIG:
    r = load(run)
    if r: curve(a, r, C340, 2.2, "-", f"340M, rank {k}  [{end(r):.3f}]", SHADE[k])
for k, run in RANKS:
    if k in (64, 128, 512):
        r = load(run)
        if r: curve(a, r, C110, 1.2, "-", f"110M, rank {k}  [{end(r):.3f}]", SHADE[k] * 0.6)
a.set_xlim(9.5e7, 3.4e8); a.set_ylim(2.6, 3.3); a.xaxis.set_major_formatter(fmt)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss (nats per token)", color=INK2)
a.set_title("340M (d1024_l24, blue) against 110M (red, faint)", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)

a = axs[2]
for runs, c, lab, mref in ((RANKS, C110, "110M", m110), (BIG, C340, "340M", m340)):
    pts = [(k, load(r)) for k, r in runs]
    pts = [(k, end(r)) for k, r in pts if r and r[2]]
    a.plot([k for k, _ in pts], [v for _, v in pts], "-o", color=c, lw=2, ms=7, label=f"{lab}, ours")
    for k, v in pts:
        a.annotate(f"{v:.3f}", (k, v), xytext=(6, 6), textcoords="offset points", fontsize=9, color=c)
    if mref and mref[2]:
        a.axhline(end(mref), color=c, lw=1.4, ls="--")
        a.annotate(f"master, {lab}: {end(mref):.3f}", (0.02, end(mref)), xycoords=("axes fraction", "data"),
                   xytext=(0, 4), textcoords="offset points", fontsize=9.5, color=c)
a.set_xscale("log", base=2); a.set_xticks([32, 64, 128, 256, 512, 1024]); a.set_xticklabels(["32", "64", "128", "256", "512", "1024\n(= full, 768)"])
a.set_xlim(24, 1500); a.set_ylim(2.65, 3.18)
a.set_xlabel("momentum rank r", color=INK2); a.set_ylabel("final validation loss", color=INK2)
a.set_title("final loss against rank (dashed = master at that size)", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=9.5, frameon=False, labelcolor=INK)
fig.suptitle("Rank and model size: the recipe at 110M and 340M, 300M tokens of Wikipedia (dashed curves = still running)",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/scale.png", dpi=95, facecolor=SURFACE)
print("wrote docs/figures/scale.png")
