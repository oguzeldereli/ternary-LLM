"""One Adam + gate variant against its references. Left: validation loss up to where the run is now (plus a margin);
right: loss difference to Adam + gate at each validation step (below zero = the variant is ahead).

  python -m scripts.plots.plot_sharp            -> docs/figures/sharp_run.png  (Adam + gate + sharp)
  python -m scripts.plots.plot_sharp undo       -> docs/figures/undo_run.png   (Adam + gate + undo)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.plots.plot_focus import load, SURFACE, INK, INK2, GRID  # noqa: E402  (also redraws focus_runs.png)

RUNS = [("master_tracked", "master weights", "#eb6834"),
        ("magadd16_wd_qk_lab", "look-ahead + sharp", "#9e9e9e"),
        ("gatevnorm_rc_s0", "Adam + gate", "#2e7d32"),
        ("gatevnorm_rc_s0_seed2", "Adam + gate, seed 2", "#81c784"),
        ("gvsharp_rc_s0", "Adam + gate + sharp", "#c2185b"),
        ("gvundo_rc_s0", "Adam + gate + undo", "#7b1fa2")]
WHICH = sys.argv[1] if len(sys.argv) > 1 else "sharp"
RUN = f"gv{WHICH}_rc_s0"
TITLE = {"sharp": "Adam + gate + sharp (additive r16 + adapter weight decay + per-head temperature), no look-ahead",
         "undo": "Adam + gate + undo (flip back last step's flips that the gradient and momentum both call uphill), no look-ahead"}[WHICH]
R = {d: load(d) for d, _, _ in RUNS}
ts, vs, _ = R[RUN]
COL = dict((d, c) for d, _, c in RUNS)[RUN]
tmax = ts[-1] * 1.15
fig, (a, b) = plt.subplots(1, 2, figsize=(18, 7), facecolor=SURFACE)
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
tg, vg, _ = R["gatevnorm_rc_s0"]
LO, GAPS = [], []
for d, lab, c in RUNS:
    if R[d] is None: continue
    t, v, _ = R[d]; m = t <= tmax
    lw = 2.8 if d == RUN else 1.8
    LO.append(v[m].min())
    a.plot(t[m], v[m], color=c, lw=lw, label=f"{lab}  [{np.interp(ts[-1], t, v):.3f} at {ts[-1] / 1e6:.0f}M]")
    if d != "gatevnorm_rc_s0":
        mm = t <= ts[-1]
        GAPS += list(v[mm] - np.interp(t[mm], tg, vg))
        b.plot(t[mm], v[mm] - np.interp(t[mm], tg, vg), color=c, lw=lw, marker="o" if d == RUN else None,
               ms=3.5, label=lab)
a.set_xlim(0, tmax); a.set_ylim(min(LO) - 0.05, min(vs[0], 5.0) + 0.05)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss", color=INK2)
a.legend(loc="upper right", fontsize=10, frameon=False, labelcolor=INK)
a.set_title(f"validation loss ({WHICH} still running)", color=INK, fontsize=11, loc="left")
b.axhline(0, color="#2e7d32", lw=1.4, ls="--")
b.set_xlim(0, ts[-1] * 1.03); b.set_ylim(min(GAPS + [0]) - 0.03, max(GAPS + [0]) + 0.03)
b.set_xlabel("training tokens", color=INK2); b.set_ylabel("loss minus Adam + gate (below 0 = better)", color=INK2)
b.annotate(f"{vs[-1] - np.interp(ts[-1], tg, vg):+.3f}", (ts[-1], vs[-1] - np.interp(ts[-1], tg, vg)), xytext=(6, -10),
           textcoords="offset points", color=COL, fontsize=11)
b.legend(loc="lower left", fontsize=10, frameon=False, labelcolor=INK)
b.set_title("gap to Adam + gate at each validation step", color=INK, fontsize=11, loc="left")
fig.suptitle(TITLE, color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"docs/figures/{WHICH}_run.png", dpi=110, facecolor=SURFACE)
print(f"wrote docs/figures/{WHICH}_run.png")
