"""LM 10M-token screens (same batches): frozen random ternary vs plain flips vs rank-256
momentum vs look-ahead variants, with master as the reference. Train loss, gain over the
frozen baseline, and final validation loss.

  python -m scripts.plots.plot_lm10m
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width, style
    ("p2_baseline",      "master weights + AdamW (reference)",  "#eb6834", 2.2, "-"),
    ("lm_frozen",        "frozen random ternary (no flips)",    "#8e8c85", 2.4, "--"),
    ("lm_plain",         "plain flips, r=0.02",                 "#1baf7a", 2.4, "-"),
    ("lm_lowrank256",    "rank-256 momentum, r=0.02",           "#7b3fb8", 2.4, "-"),
    ("lm_lowrank256_r04", "rank-256 momentum, r=0.04",          "#b39ddb", 1.8, "-"),
    ("lm_lowrank256_r1", "rank-256 momentum, r=0.1 (stopped)",  "#d1c4e9", 1.6, "-"),
    ("lm_lowrank256_adapt", "rank-256 momentum, decay 0.97*cos(g,M)", "#e0a100", 2.4, "-"),
    ("lm_lowrank256_adapt_xb2", "adaptive momentum + cross-batch look-ahead x2 (stopped)", "#00897b", 2.8, "-"),
    ("lm_lowrank256_xb2", "rank-256 momentum (fixed 0.97) + cross-batch look-ahead x2", "#e53935", 2.8, "-"),
    ("la_fast_fp32tail", "same-batch look-ahead",               "#c2185b", 2.4, "-"),
    ("la_xb2",           "cross-batch look-ahead x2",           "#111111", 2.6, "-"),
    ("la_xb2_rc",        "cross-batch look-ahead x2 + row/col scales", "#2a78d6", 2.2, ":"),
]
MAX_TOK = 10.4e6


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r for r in rows if "loss" in r and "tokens" in r and r["tokens"] <= MAX_TOK}
    s = sorted(tr)
    t = np.array([tr[i]["tokens"] for i in s], float)
    L = np.array([tr[i]["loss"] for i in s])
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(15, i // 5 + 1) for i in range(len(L))]
    Ls = np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])
    v = [r for r in rows if r.get("final") and "val_loss" in r]
    c = [(r["step"], r["lr_cos"]) for r in rows if "lr_cos" in r]
    return t, Ls, (v[-1]["val_loss"] if v and d != "p2_baseline" else None), c


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/lm10m.png")
a = ap.parse_args()
D = {d: load(d) for d, *_ in RUNS if os.path.exists(f"checkpoints/{d}/metrics.jsonl")}
fig, axs = plt.subplots(2, 2, figsize=(20, 13), facecolor=SURFACE)
(ax1, ax2), (ax4, ax3) = axs
tf, Lf, *_ = D["lm_frozen"]
for d, lab, col, lw, ls in RUNS:
    if d not in D: continue
    t, L, _, c = D[d]
    ax1.plot(t, L, color=col, lw=lw, ls=ls, label=lab)
    if d != "lm_frozen":
        ax2.plot(t, np.interp(t, tf, Lf) - L, color=col, lw=lw, ls=ls, label=lab)
    if c:
        st = np.array([a for a, _ in c]); cv = np.array([b for _, b in c])
        ax4.plot(st * 32768, cv, color=col, lw=1.8, label=lab)
ax2.axhline(0, color="#8e8c85", lw=2.0, ls="--")
ax4.axhline(0, color=INK2, lw=1)
vals = [(lab, col, D[d][2]) for d, lab, col, *_ in RUNS if d in D and D[d][2] is not None]
vals.sort(key=lambda r: -r[2])
y = np.arange(len(vals))
ax3.barh(y, [v for *_, v in vals], color=[c for _, c, _ in vals], height=0.62)
for i, (_, _, v) in enumerate(vals):
    ax3.text(v + 0.02, i, f"{v:.3f}  (ppl {np.exp(v):.0f})", va="center", color=INK, fontsize=10)
ax3.set_yticks(y); ax3.set_yticklabels([l for l, *_ in vals], fontsize=9.5)
ax3.set_xlim(4.6, 6.5)
ax1.set_ylim(4.6, 11); ax2.set_ylim(-0.8, 3.2)
titles = ("Train loss (smoothed), same batches",
          "Gain over frozen random ternary (frozen loss - run loss)",
          "Validation loss at 10M tokens",
          "cos(gradient, low-rank momentum), mean over 84 layers")
for ax, tt in zip((ax1, ax2, ax3, ax4), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
for ax in (ax1, ax2, ax4):
    ax.set_xscale("log"); ax.set_xlabel("training tokens", color=INK2)
    ax.set_xlim(3e4, MAX_TOK)
ax1.set_yscale("log"); ax1.legend(fontsize=9, frameon=False)
ax2.legend(fontsize=9, frameon=False, loc="upper right")
ax4.set_ylim(-1, 1); ax4.legend(fontsize=9, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
