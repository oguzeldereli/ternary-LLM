"""Tail learning-rate control: does the flip arms' 5x lower tail LR explain the gap?

  python3 plot_lr_control.py --out docs/lr_control.png
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BASE, FLIP, CTL = "#eb6834", "#2a78d6", "#1baf7a"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="docs/figures/lr_control.png")
ap.add_argument("--steps", type=int, default=1050)
args = ap.parse_args()


def loss_curve(run, upto, win=25):
    r = [json.loads(l) for l in open(f"checkpoints/{run}/metrics.jsonl")
         if '"loss"' in l and '"val_loss"' not in l]
    s = np.array([x["step"] for x in r]); v = np.array([x["loss"] for x in r])
    m = s < upto; s, v = s[m], v[m]
    k = np.ones(win) / win
    sm = np.convolve(v, k, "valid")
    return s[win - 1:], sm


series = [("p2_baseline",     "master weights, tail LR 1.5e-3", BASE, "-"),
          ("armA_cosine",     "flips, tail LR 3e-4  (all published arms)", FLIP, "-"),
          ("lr_ctl",          "flips, tail LR 1.5e-3  (control)", CTL, "-"),
          ("armA_cos_lr15",   "same, independent repeat", CTL, (0, (5, 2)))]

fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True,
                         gridspec_kw=dict(hspace=0.16, height_ratios=[1.35, 1]))
fig.patch.set_facecolor(SURFACE)


def style(ax, ylabel):
    ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


ax = axes[0]; style(ax, "training loss (nats, 25-step mean)")
curves = {}
for run, label, col, ls in series:
    s, v = loss_curve(run, args.steps)
    curves[run] = (s, v)
    ax.plot(s, v, color=col, linewidth=2, linestyle=ls)
    ax.plot([], [], color=col, linewidth=2, linestyle=ls, label=label)
ax.set_xlim(0, args.steps * 1.02)
ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper right",
          borderaxespad=0.6)
ax.set_title("Same module, two learning rates: the float tail (embedding + norms)\n"
             "110M · seq 2048 · 32,768 tokens/step · everything else identical",
             color=INK, fontsize=12, loc="left", pad=10)

# shade where the control is ahead of the baseline
sb, vb = curves["p2_baseline"]; sc, vc = curves["lr_ctl"]
vb_i = np.interp(sc, sb, vb)
ahead = sc[vc < vb_i]
if len(ahead):
    ax.axvspan(ahead.min(), ahead.max(), color=CTL, alpha=0.10, zorder=0)
    ax.annotate("control is ahead of the\nmaster-weight baseline here",
                (ahead.max(), vc.max()), xytext=(10, -6), textcoords="offset points",
                fontsize=8.5, color=CTL)

ax = axes[1]; style(ax, "gap to baseline (nats)")
sf, vf = curves["armA_cosine"]
for run, label, col, ls in [("armA_cosine", "tail LR 3e-4", FLIP, "-"),
                            ("lr_ctl", "tail LR 1.5e-3", CTL, "-"),
                            ("armA_cos_lr15", "", CTL, (0, (5, 2)))]:
    s, v = curves[run]
    ax.plot(s, v - np.interp(s, sb, vb), color=col, linewidth=2, linestyle=ls)
    if not label:
        continue
    ax.annotate(label, (s[-1], (v - np.interp(s, sb, vb))[-1]), xytext=(-4, -14),
                textcoords="offset points", va="center", ha="right", fontsize=9,
                color=col, fontweight="bold")
ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
closed = 1 - (vc[-1] - vb_i[-1]) / (np.interp(sc[-1], sf, vf) - vb_i[-1])
ax.annotate(f"matching the tail LR closes only {closed*100:.0f}% of the gap by step {args.steps}\n"
            f"— the other {100-closed*100:.0f}% is the update rule.  val ppl at step 1000:\n"
            f"188.3 (3e-4)  vs  153.3 (1.5e-3)",
            (0.03, 0.97), xycoords="axes fraction", fontsize=9.5, color=INK2,
            style="italic", ha="left", va="top")
ax.set_xlabel("step", color=INK2, fontsize=10)

fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(args.out)
