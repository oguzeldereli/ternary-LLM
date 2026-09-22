"""Log-log view: do the flip runs scale like the baseline, or diverge from it?

Fits L(D) = A * D^(-alpha) over the tail of each run (loss in nats vs tokens).
Parallel lines = a fixed multiplicative penalty; different slopes = different
scaling exponents.
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/scaling.png")
args = ap.parse_args()

RUNS = [("p2_baseline",   "master weights",                "#eb6834"),
        ("armA_cos_600M", "flips, annealed (600M)",        "#1baf7a"),
        ("armA_cosine",   "flips, annealed (300M)",        "#2a78d6"),
        ("p3b_acc1",      "flips, constant rate",          "#4a3aa7")]


def curve(run, win=51):
    r = [json.loads(l) for l in open(f"checkpoints/{run}/metrics.jsonl")
         if '"loss"' in l and '"val_loss"' not in l]
    d = np.array([x["tokens"] for x in r], float)
    v = np.array([x["loss"] for x in r], float)
    k = np.ones(win) / win
    return d[win - 1:], np.convolve(v, k, "valid")


fig, ax = plt.subplots(figsize=(9.5, 6.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
ax.grid(True, which="both", color=GRID, linewidth=0.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d3")
ax.tick_params(colors=INK2, labelsize=9)

for run, label, col in RUNS:
    d, v = curve(run)
    ax.plot(d / 1e6, v, color=col, linewidth=2, label=label)
    m = d > d.max() * 0.25                      # fit the last 3/4, after warmup
    a, b = np.polyfit(np.log(d[m]), np.log(v[m]), 1)
    ax.plot(d[m] / 1e6, np.exp(b) * d[m] ** a, color=col, linewidth=1,
            linestyle=(0, (3, 3)), alpha=0.9)
    print(f"{label:<26} slope (alpha) {a:+.4f}   loss at end {v[-1]:.3f}")
    ax.annotate(f"α = {a:+.3f}", (d[-1] / 1e6, v[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=col,
                fontweight="bold")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("tokens (M, log)", color=INK2, fontsize=10)
ax.set_ylabel("training loss, nats (log)", color=INK2, fontsize=10)
ax.set_xlim(3, 1500)
ax.set_title("Log-log: the flip runs are 2.4-11x flatter than the baseline\n"
             "dashed = L = A*D^-alpha fit over the last three quarters (LOCAL slope, not asymptotic)",
             color=INK, fontsize=12, loc="left", pad=10)
ax.annotate("a 3-parameter fit L = L_inf + A*D^-alpha is degenerate over this range:\n"
            "the two annealed runs (same recipe) give L_inf = 0.00 and 3.34.\n"
            "Floor and exponent cannot be separated with <1 decade of tokens.",
            (0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.5, color=INK2, style="italic")
ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower left")
fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(args.out)
