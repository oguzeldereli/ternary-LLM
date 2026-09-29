"""The best run of each family, nothing else: validation loss against training tokens. Left: the whole run (log
tokens); right: the last stretch (100M-300M, linear) where the families separate. Running runs are dashed.

  python -m scripts.plots.plot_best      -> docs/figures/best_runs.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # (run, label, colour)
    ("fp32_baseline", "full precision fp32 + AdamW (reference)", "#9e9e9e"),
    ("master_tracked", "master weights: ternary forward, fp32 latent (reference)", "#eb6834"),
    ("magadd16_wd_qk_lab", "best with look-ahead: + additive r16, adapter weight decay, head temperature", "#fbc02d"),
    ("lm_lowrank256_xb2_100M", "momentum + look-ahead (old baseline)", "#111111"),
    ("gatevnorm_rc_s0", "best without look-ahead: sign gate + Adam-style step + row/col scales", "#2e7d32"),
    ("vnorm_rc_s0", "Adam-style step + row/col scales", "#00897b"),
    ("rc_s0", "plain momentum + row/col scales", "#1565c0"),
    ("nola_lab", "plain momentum, no look-ahead, no scales", "#7b1fa2"),
    ("user_cap3_s0", "your rule: no friction, velocity cap 3, absolute flip chance", "#c2185b"),
    ("user_cap1_s0", "your rule: velocity cap 1", "#ef6c00"),
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
    lw = 2.6 if d in ("gatevnorm_rc_s0", "magadd16_wd_qk_lab") else 1.8
    a.plot(t, v, ls, color=c, lw=lw, label=f"{lab}  [{tag}]")
    m = t >= 9.5e7
    b.plot(t[m], v[m], ls, color=c, lw=lw)
    if m.any():   # end labels, nudged apart when two runs end within 0.012 of each other
        y = v[m][-1]
        while any(abs(y - u) < 0.012 for u in ENDS): y += 0.012
        ENDS.append(y)
        b.annotate(f"{v[m][-1]:.3f}", (t[m][-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                   fontsize=10, color=c)
a.set_xscale("log"); a.set_ylim(2.6, 5.0); a.set_xlim(2e7, 3.2e8)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss", color=INK2)
b.set_xlim(9.5e7, 3.25e8); b.set_ylim(2.6, 3.6)
b.set_xlabel("training tokens (100M-300M)", color=INK2)
b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
a.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
fig.suptitle("Best run of each family (validation loss; dashed = still running)", color=INK, fontsize=14, x=0.02,
             ha="left")
fig.tight_layout()
fig.savefig("docs/figures/best_runs.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/best_runs.png")
