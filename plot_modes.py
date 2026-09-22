"""One figure: final perplexity of every training mode, plus their trajectories.

  python3 plot_modes.py --out docs/modes.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CEILING, FLIP, BEST = "#eb6834", "#2a78d6", "#1baf7a"     # palette slots 2, 1, 3

# (run dir, label, tokens, family)
RUNS = [
    ("p2_baseline",       "master weights + STE + AdamW",      "300M", "ceiling"),
    ("armA_cos_600M",     "flips, annealed rate",              "600M", "best"),
    ("armA_cosine",       "flips, annealed rate (cosine)",     "300M", "best"),
    ("armA_linear",       "flips, annealed rate (linear)",     "300M", "flip"),
    ("armA_cos_lockout",  "flips, annealed + per-weight lockout", "300M", "flip"),
    ("armA_cos_floor",    "flips, annealed to a 0.02% floor",  "300M", "flip"),
    ("p3b_acc4",          "flips, constant rate, 4x batch",    "400M", "flip"),
    ("p3b_acc1",          "flips, constant rate",              "300M", "flip"),
    ("armA_cos_ef",       "flips, annealed + error feedback",  "300M", "flip"),
    ("armB_absscale",     "flips, frozen threshold scale",     "300M", "flip"),
]
COLOR = {"ceiling": CEILING, "best": BEST, "flip": FLIP}

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/modes.png")
args = ap.parse_args()


def series(run):
    p = f"checkpoints/{run}/metrics.jsonl"
    recs = [json.loads(l) for l in open(p)]
    tr = [r for r in recs if "tokens" in r]
    va = [r for r in recs if "val_ppl" in r]
    steps = np.array([r["step"] for r in tr]); toks = np.array([r["tokens"] for r in tr]) / 1e6
    vs = np.array([v["step"] for v in va]); vp = np.array([v["val_ppl"] for v in va])
    return np.interp(vs, steps, toks), vp


rows = []
for run, label, tok, fam in RUNS:
    if os.path.exists(f"checkpoints/{run}/metrics.jsonl"):
        t, p = series(run)
        rows.append((label, tok, fam, float(p[-1]), t, p))
rows.sort(key=lambda r: r[3])

fig = plt.figure(figsize=(11.5, 9.5), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.26)


def style(ax):
    ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#d9d8d3")
    ax.tick_params(colors=INK2, labelsize=9)


ax = fig.add_subplot(gs[0]); style(ax)
y = np.arange(len(rows))
ax.barh(y, [r[3] for r in rows], color=[COLOR[r[2]] for r in rows], height=0.62)
ax.set_yticks(y, [f"{r[0]}  ·  {r[1]}" for r in rows], fontsize=9.5, color=INK)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(10, max(r[3] for r in rows) * 1.9)
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
for i, r in enumerate(rows):
    ax.annotate(f"{r[3]:.2f}", (r[3], i), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
ax.set_xlabel("held-out perplexity (log scale, lower is better)", color=INK2, fontsize=10)
ax.set_title("110M params · seq 2048 · 32,768 tokens/step · Wikipedia (Llama 32k)\n"
             "identical in every respect except the weight-update rule",
             color=INK, fontsize=12.5, loc="left", pad=12)
ax.annotate(f"best master-free result costs {rows[1][3] / rows[0][3]:.1f}x the ceiling",
            (0.99, 0.93), xycoords="axes fraction", ha="right", fontsize=10,
            color=INK2, style="italic")

ax = fig.add_subplot(gs[1]); style(ax)
ax.grid(True, color=GRID, linewidth=0.8)
for label, tok, fam, final, t, p in rows:
    ax.plot(t, p, color=COLOR[fam], linewidth=2.0 if fam != "flip" else 1.4,
            alpha=1.0 if fam != "flip" else 0.5, zorder=3 if fam != "flip" else 2)
for lab, xy, col in [("master weights", (300, 15.79), CEILING),
                     ("best master-free (600M tokens)", (600, 92.72), BEST),
                     ("other flip rules", (300, 160.68), FLIP)]:
    ax.annotate(lab, xy, xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=col, fontweight="bold")
ax.set_yscale("log"); ax.set_ylabel("held-out perplexity", color=INK2, fontsize=10)
ax.set_xlabel("tokens (M)", color=INK2, fontsize=10)
ax.set_title("how they got there", color=INK2, fontsize=10, loc="left", pad=8)
ax.annotate("mid-training points are single noisy evals (see RUNS.md)",
            (0.99, 0.95), xycoords="axes fraction", ha="right", fontsize=8.5,
            color=INK2, style="italic")

fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(args.out)
