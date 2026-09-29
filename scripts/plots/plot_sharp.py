"""Adam + gate + sharp against its references. Left: validation loss up to where sharp is now (plus a margin);
right: loss difference to Adam + gate at each validation step (below zero = sharp ahead).

  python -m scripts.plots.plot_sharp      -> docs/figures/sharp_run.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.plots.plot_focus import load, SURFACE, INK, INK2, GRID  # noqa: E402  (also redraws focus_runs.png)

RUNS = [("master_tracked", "master weights", "#eb6834"),
        ("magadd16_wd_qk_lab", "look-ahead + sharp", "#9e9e9e"),
        ("gatevnorm_rc_s0", "Adam + gate", "#2e7d32"),
        ("gatevnorm_rc_s0_seed2", "Adam + gate, seed 2", "#81c784"),
        ("gvsharp_rc_s0", "Adam + gate + sharp", "#c2185b")]
R = {d: load(d) for d, _, _ in RUNS}
ts, vs, _ = R["gvsharp_rc_s0"]
tmax = ts[-1] * 1.15
fig, (a, b) = plt.subplots(1, 2, figsize=(18, 7), facecolor=SURFACE)
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
tg, vg, _ = R["gatevnorm_rc_s0"]
for d, lab, c in RUNS:
    if R[d] is None: continue
    t, v, _ = R[d]; m = t <= tmax
    lw = 2.8 if d == "gvsharp_rc_s0" else 1.8
    a.plot(t[m], v[m], color=c, lw=lw, label=f"{lab}  [{np.interp(ts[-1], t, v):.3f} at {ts[-1] / 1e6:.0f}M]")
    if d != "gatevnorm_rc_s0":
        mm = t <= ts[-1]
        b.plot(t[mm], v[mm] - np.interp(t[mm], tg, vg), color=c, lw=lw, marker="o" if d == "gvsharp_rc_s0" else None,
               ms=3.5, label=lab)
a.set_xlim(0, tmax); a.set_ylim(3.0, 4.6)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss", color=INK2)
a.legend(loc="upper right", fontsize=10, frameon=False, labelcolor=INK)
a.set_title("validation loss (sharp still running)", color=INK, fontsize=11, loc="left")
b.axhline(0, color="#2e7d32", lw=1.4, ls="--")
b.set_xlim(0, ts[-1] * 1.03); b.set_ylim(-0.35, 0.1)
b.set_xlabel("training tokens", color=INK2); b.set_ylabel("loss minus Adam + gate (below 0 = better)", color=INK2)
b.annotate(f"{vs[-1] - np.interp(ts[-1], tg, vg):+.3f}", (ts[-1], vs[-1] - np.interp(ts[-1], tg, vg)), xytext=(6, -10),
           textcoords="offset points", color="#c2185b", fontsize=11)
b.legend(loc="lower left", fontsize=10, frameon=False, labelcolor=INK)
b.set_title("gap to Adam + gate at each validation step", color=INK, fontsize=11, loc="left")
fig.suptitle("Adam + gate + sharp (additive r16 + adapter weight decay + per-head temperature), no look-ahead",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/sharp_run.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/sharp_run.png")
