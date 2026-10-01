"""The three knobs of the current recipe, final validation loss at 300M (110M model, sharp base: gate + Adam step +
additive r16 + adapter weight decay + head temperature). Left: momentum decay (as its memory 1/(1-b)); middle: dry
friction strength D (with decay 1); right: momentum rank, for each memory setting. Finals are read from the run logs.

  python -m scripts.plots.plot_sweeps      -> docs/figures/sweeps.png
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MASTER = 2.7510


def fin(d):
    for c in (f"checkpoints/{d}_lab/train.log", f"checkpoints/{d}/train.log"):
        if os.path.exists(c):
            m = re.findall(r"FINAL val loss ([\d.]+)", open(c).read())
            if m: return float(m[-1])
    return None


DECAY = [(0.97, "gvsharp_rc_s0"), (0.99, "gvsharp_b099_s0"), (0.995, "gvsharp_b0995_s0"), (0.998, "gvsharp_b0998_s0"),
         (1.0, "gvsharp_b1_s0")]
DRY = [(0.01, "gvsharp_dry01_s0"), (0.02, "gvsharp_dry02_s0"), (0.0303, "gvsharp_dry_s0"), (0.05, "gvsharp_dry05_s0"),
       (0.1, "gvsharp_dry1_s0")]
RANK = [  # (family label, colour, [(rank, run)]); value labels offset per family so they do not collide
    ("short memory (decay 0.97)", "#c2185b", (8, 4), [(256, "gvsharp_rc_s0"), (512, "gvsharp_r512_s0")]),
    ("decay 0.995", "#1e88e5", (-44, 6), [(256, "gvsharp_b0995_s0"), (512, "gvsharp_b0995_r512_s0")]),
    ("dry friction", "#ef6c00", (8, 6), [(256, "gvsharp_dry_s0"), (512, "gvsharp_dry_r512_s0"), (1024, "gvsharp_dry_r1024_s0")]),
    ("dry friction + spend", "#b71c1c", (8, -14), [(32, "rk32_dryspend_s0"), (64, "rk64_dryspend_s0"),
                                                   (128, "rk128_dryspend_s0"), (256, "gvsharp_dryspend_s0"),
                                                   (512, "gvsharp_dryspend_r512_s0")]),
]

fig, axs = plt.subplots(1, 3, figsize=(22, 7.2), facecolor=SURFACE)
for ax in axs:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.axhline(MASTER, color="#eb6834", lw=1.4, ls="--")
    ax.set_ylim(2.72, 3.08)
axs[0].annotate("master weights 2.751", (0.02, MASTER + 0.004), xycoords=("axes fraction", "data"), color="#eb6834",
                fontsize=10)

a = axs[0]
pts = [(b, fin(r)) for b, r in DECAY if fin(r) is not None]
mem = lambda b: 1 / (1 - b) if b < 1 else 3000.0          # decay 1 = no forgetting, drawn at the right edge
a.plot([mem(b) for b, _ in pts], [v for _, v in pts], "-o", color="#1e88e5", lw=2, ms=8)
for b, v in pts:
    a.annotate(f"b={b:g}\n{v:.3f}", (mem(b), v), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5,
               color=INK2)
a.set_xscale("log"); a.set_xlim(20, 5000)
a.set_xticks([33, 100, 200, 500, 3000]); a.set_xticklabels(["33", "100", "200", "500", "no decay"])
a.set_xlabel("momentum memory, steps = 1 / (1 - decay)", color=INK2); a.set_ylabel("final validation loss", color=INK2)
a.set_title("decay: best at 0.995 (memory ~200 steps)", color=INK, fontsize=11, loc="left")

a = axs[1]
pts = [(D, fin(r)) for D, r in DRY if fin(r) is not None]
a.plot([D for D, _ in pts], [v for _, v in pts], "-o", color="#ef6c00", lw=2, ms=8)
for D, v in pts:
    a.annotate(f"{v:.3f}", (D, v), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5, color=INK2)
a.set_xscale("log"); a.set_xlim(0.007, 0.14)
a.set_xticks([0.01, 0.02, 0.0303, 0.05, 0.1]); a.set_xticklabels(["0.01", "0.02", "1/33", "0.05", "0.1"])
a.set_xlabel("dry friction D (fraction of the gradient norm removed per step); weaker = longer memory", color=INK2)
a.set_title("dry friction: best at 1/33-0.05", color=INK, fontsize=11, loc="left")

a = axs[2]
missing = []
for lab, c, off, pr in RANK:
    pts = [(k, fin(r)) for k, r in pr if fin(r) is not None]
    missing += [k for k, r in pr if fin(r) is None]
    if not pts: continue
    a.plot([k for k, _ in pts], [v for _, v in pts], "-o", color=c, lw=2, ms=7, label=lab)
    for k, v in pts:
        a.annotate(f"{v:.3f}", (k, v), xytext=off, textcoords="offset points", fontsize=9, color=c)
if missing:
    a.annotate("dry friction + spend at rank " + " / ".join(str(k) for k in sorted(missing)) + ": still training",
               (0.03, 0.04), xycoords="axes fraction", fontsize=9.5, color=INK2)
a.set_xscale("log", base=2); a.set_xlim(24, 1500)
a.set_xticks([32, 64, 128, 256, 512, 1024]); a.set_xticklabels(["32", "64", "128", "256", "512", "1024\n(= full, 768)"])
a.set_xlabel("momentum rank r (state = r x (rows + columns) per layer)", color=INK2)
a.set_ylim(2.72, 3.18)   # own range: rank 32 sits at 3.14
a.set_title("rank: each halving below 512 costs 0.05-0.10; 1024 adds 0.016", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=9.5, frameon=False, labelcolor=INK)
fig.suptitle("The recipe's three knobs: final validation loss at 300M tokens (110M model; dashed = master weights)",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/sweeps.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/sweeps.png")
