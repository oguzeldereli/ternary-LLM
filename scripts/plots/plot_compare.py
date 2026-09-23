"""Overlay runs: val perplexity and flip rate vs tokens.

  python3 plot_compare.py checkpoints/p3b_acc1 checkpoints/armB_absscale --out cmp.png

Flip rate is the primary evidence for the Arm B hypothesis: a frozen absolute
threshold should make it DECAY on its own. Arm A decays by construction, so only
its perplexity is informative.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="checkpoints/compare.png")
ap.add_argument("--title", default="stateless 1.58-bit: flip rule variants")
args = ap.parse_args()

fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                         gridspec_kw=dict(hspace=0.25))
fig.patch.set_facecolor(SURFACE)


def style(ax, ylabel, logy=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if logy:
        ax.set_yscale("log")


for i, run in enumerate(args.runs):
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        continue
    recs = [json.loads(l) for l in open(path)]
    tr = [r for r in recs if "flip_frac" in r]
    va = [r for r in recs if "val_ppl" in r]
    name = os.path.basename(run.rstrip("/"))
    col = SERIES[i % len(SERIES)]
    if tr:
        tok = np.array([r["tokens"] for r in tr]) / 1e6
        steps = np.array([r["step"] for r in tr])
        if va:
            vs = np.array([v["step"] for v in va])
            vp = np.array([v["val_ppl"] for v in va])
            vt = np.interp(vs, steps, tok)
            axes[0].plot(vt, vp, color=col, linewidth=2, marker="o", markersize=4,
                         label=name)
            axes[0].annotate(f"{vp[-1]:.0f}", (vt[-1], vp[-1]), color=INK, fontsize=9,
                             xytext=(4, 0), textcoords="offset points")
        ff = np.array([r["flip_frac"] for r in tr]) * 100
        k = max(1, len(ff) // 300)
        axes[1].plot(tok[::k], np.where(ff > 0, ff, np.nan)[::k], color=col, linewidth=2,
                     label=name)
        nf = np.array([r["never_frac"] for r in tr]) * 100
        axes[2].plot(tok[::k], nf[::k], color=col, linewidth=2, label=name)

style(axes[0], "val perplexity", logy=True)
axes[0].set_title(args.title, color=INK, fontsize=12, loc="left", pad=10)
axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
style(axes[1], "flips / step (% of weights)", logy=True)
axes[1].set_title("does flipping stop on its own?", color=INK2, fontsize=10, loc="left")
style(axes[2], "never flipped (%)")
axes[2].set_title("weights that have never moved", color=INK2, fontsize=10, loc="left")
axes[-1].set_xlabel("tokens (M)", color=INK2, fontsize=10)
fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(args.out)
