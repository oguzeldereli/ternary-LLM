"""Today's runs (29 Sep) on a linear token axis, 0-300M: validation loss. Left: the whole run; right: 100M-300M.
Running runs are dashed.

  python -m scripts.plots.plot_focus      -> docs/figures/focus_runs.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # (run, label, colour)
    ("master_tracked", "master weights (reference)", "#eb6834"),
    ("magadd16_wd_qk_lab", "look-ahead + sharp (best with look-ahead)", "#9e9e9e"),
    ("rc_s0", "plain momentum + row/col scales", "#1565c0"),
    ("gatevnorm_rc_s0", "Adam + gate", "#2e7d32"),
    ("gatevnorm_rc_s0_seed2", "Adam + gate, seed 2", "#81c784"),
    ("gvsharp_rc_s0", "Adam + gate + sharp (r16 + wd + head temperature)", "#c2185b"),
    ("gvundo_rc_s0", "Adam + gate + undo (stopped)", "#7b1fa2"),
    ("gvsharp_b099_s0", "sharp + beta 0.99", "#1e88e5"),
    ("gvsharp_b1_s0", "sharp + beta 1", "#5e35b1"),
    ("gvsharp_r512_s0", "sharp + rank 512", "#6d4c41"),
    ("gvsharp_slowgate_s0", "sharp + slow gate (r64, beta 0.999)", "#00897b"),
    ("gvsharp_dither_s0", "sharp + low-discrepancy dither", "#bdbdbd"),
    ("gvsharp_dry_s0", "sharp + beta 1 + dry friction (vector)", "#f9a825"),
    ("gvsharp_b1spend_s0", "sharp + beta 1 + spend 3", "#212121"),
]


def load(d):
    if os.path.exists(f"checkpoints/{d}_lab/metrics.jsonl"): d = f"{d}_lab"
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    va, tps, last = {}, 32768.0, 0
    for l in open(f):
        r = json.loads(l)
        if "loss" in r and "tokens" in r: tps = r["tokens"] / (r["step"] + 1); last = max(last, r["step"])
        if "val_loss" in r: va[r["step"]] = r["val_loss"]
    if not va: return None
    s = np.array(sorted(va))
    return (s + 1) * tps, np.array([va[i] for i in s]), last >= 9150


fig, (a, b) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]})
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
ENDS = []
for d, lab, c in sorted(RUNS, key=lambda x: (load(x[0]) or (0, [9]))[1][-1]):
    r = load(d)
    if r is None: continue
    t, v, done = r
    ls = "-" if done else "--"
    tag = f"{v[-1]:.3f}" + ("" if done else f" at {t[-1] / 1e6:.0f}M, running")
    lw = 2.8 if d.startswith("gvsharp") else 1.8
    a.plot(t, v, ls, color=c, lw=lw, label=f"{lab}  [{tag}]")
    m = t >= 9.5e7
    b.plot(t[m], v[m], ls, color=c, lw=lw)
    if m.any():   # end labels, nudged apart when two runs end within 0.012 of each other
        y = v[m][-1]
        while any(abs(y - u) < 0.012 for u in ENDS): y += 0.012
        ENDS.append(y)
        b.annotate(f"{v[m][-1]:.3f}", (t[m][-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                   fontsize=10, color=c)
a.set_ylim(2.6, 5.0); a.set_xlim(0, 3.1e8)
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
a.set_xlabel("training tokens (linear, 0-300M)", color=INK2); a.set_ylabel("validation loss", color=INK2)
b.set_xlim(9.5e7, 3.25e8); b.set_ylim(2.6, 3.6)
b.set_xlabel("training tokens (100M-300M)", color=INK2)
b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
a.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
fig.suptitle("Focus: Adam + gate + sharp and the night 29-30 Sep variants (validation loss, linear tokens; dashed = still running)", color=INK, fontsize=14, x=0.02,
             ha="left")
fig.tight_layout()
fig.savefig("docs/figures/focus_runs.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/focus_runs.png")
