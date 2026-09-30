"""Every finished run of the night 29-30 Sep (all from scratch, 300M, no look-ahead, base = Adam + gate + sharp) with
the references. Left: validation loss 100M-300M; right: final validation loss, sorted, as the gap to master.

  python -m scripts.plots.plot_night      -> docs/figures/night_runs.png
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.plots.plot_focus import load, SURFACE, INK, INK2, GRID  # noqa: E402  (also redraws focus_runs.png)

REF = [("master_tracked", "master weights (reference)", "#eb6834"),
       ("magadd16_wd_qk_lab", "look-ahead + sharp", "#9e9e9e"),
       ("gatevnorm_rc_s0", "Adam + gate", "#2e7d32"),
       ("gvsharp_rc_s0", "Adam + gate + sharp (the base)", "#c2185b")]
RUNS = [  # (run, label, family colour): blue = decay, orange = dry friction, grey = other
    ("gvsharp_b099_s0", "beta 0.99", "#64b5f6"), ("gvsharp_b0995_s0", "beta 0.995", "#1565c0"),
    ("gvsharp_b1_s0", "beta 1 (no friction)", "#90a4ae"),
    ("gvsharp_b099spend_s0", "beta 0.99 + spend", "#42a5f5"), ("gvsharp_b099slow_s0", "beta 0.99 + slow gate", "#90caf9"),
    ("gvsharp_dry_s0", "dry friction 0.0303", "#ef6c00"), ("gvsharp_dry01_s0", "dry friction 0.01", "#ffcc80"),
    ("gvsharp_dry1_s0", "dry friction 0.1", "#ffe0b2"), ("gvsharp_dryspend_s0", "dry friction + spend", "#e65100"),
    ("gvsharp_dryw_s0", "dry friction per weight", "#ffb74d"), ("gvsharp_dry_r512_s0", "dry friction + rank 512", "#bf360c"),
    ("gvsharp_b1spend_s0", "beta 1 + spend", "#78909c"), ("gvsharp_b1spend_slowgate_s0", "beta 1 + spend + slow gate", "#b0bec5"),
    ("gvsharp_slowgate_s0", "slow gate", "#546e7a"), ("gvsharp_r512_s0", "rank 512", "#37474f"),
    ("gvsharp_dither_s0", "low-discrepancy dither", "#cfd8dc"),
    ("gvundo_rc_s0", "undo (stopped at 66M)", "#7b1fa2")]
MASTER = 2.7510


def final(d):
    for c in (f"checkpoints/{d}_lab/train.log", f"checkpoints/{d}/train.log"):
        if os.path.exists(c):
            f = re.findall(r"FINAL val loss ([\d.]+)", open(c).read())
            if f: return float(f[-1])
    return None


fig, (a, b) = plt.subplots(1, 2, figsize=(22, 10), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
bars = []
for d, lab, c in REF + RUNS:
    r = load(d)
    if r is None: continue
    t, v, done = r
    fv = final(d) if done else None
    if not done and d != "gvundo_rc_s0": continue          # finished runs only
    ref = d in dict((x[0], 1) for x in REF)
    m = t >= 9.5e7
    if m.any():
        a.plot(t[m], v[m], color=c, lw=2.6 if ref else 1.7, ls="--" if ref else "-")
    bars.append((lab, c, fv if fv is not None else v[-1], ref, done))
a.set_xlim(9.5e7, 3.05e8); a.set_ylim(2.72, 3.45)
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss", color=INK2)
a.set_title("validation loss, 100M-300M (dashed = references; blue = decay family, orange = dry friction)", color=INK,
            fontsize=11, loc="left")
bars = [x for x in bars if x[4]]
bars.sort(key=lambda x: x[2])
y = np.arange(len(bars))
b.barh(y, [v - MASTER for _, _, v, _, _ in bars], color=[c for _, c, *_ in bars],
       edgecolor=[INK if r else "none" for *_, r, _ in bars])
b.set_yticks(y); b.set_yticklabels([lab for lab, *_ in bars], fontsize=10); b.invert_yaxis()
for i, (_, _, v, _, _) in enumerate(bars):
    b.annotate(f"{v:.4f}  (+{v - MASTER:.3f})", (v - MASTER, i), xytext=(4, 0), textcoords="offset points",
               va="center", fontsize=9, color=INK2)
b.set_xlim(0, 0.37)
b.set_xlabel("final validation loss minus master (2.751)", color=INK2)
b.set_title("finished runs, final validation loss (300M tokens)", color=INK, fontsize=11, loc="left")
fig.suptitle("Night 29-30 Sep: memory length is the lever (all from scratch, 300M, no look-ahead, base = Adam + gate + "
             "sharp)", color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/night_runs.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/night_runs.png", len(bars), "bars")
