"""The 340M model (d1024_l24, 300M tokens): validation loss of master weights and of our recipe (dry friction + spend) at
rank 64 / 128 / 512 / full (1024), and each run's gap to master over training.

  python -m scripts.plots.plot_340m      -> docs/figures/runs_340m.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.plots.plot_best import load, SURFACE, INK, INK2, GRID

RUNS = [("big_master", "master weights (fp32 latent, AdamW)", "#eb6834", 2.6),
        ("big_dryspend_r1024_s0", "ours, full-rank momentum (1024)", "#0d1b4c", 2.4),
        ("big_dryspend_r512_s0", "ours, rank 512 (half of full)", "#1565c0", 2.0),
        ("big_rk128_dryspend_s0", "ours, rank 128", "#64b5f6", 1.8),
        ("big_rk64_dryspend_s0", "ours, rank 64", "#bbdefb", 1.8)]
fig, (a, b) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor=SURFACE)
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
D = {r: load(r) for r, *_ in RUNS}
tm, vm, _, fm = D["big_master"]
for r, lab, c, lw in RUNS:
    if D[r] is None: continue
    t, v, done, fin = D[r]
    e = fin if fin is not None else v[-1]
    m = t >= 2e7
    a.plot(t[m], v[m], color=c, lw=lw, label=f"{lab}  [{e:.4f}, ppl {np.exp(e):.1f}]")
    if r != "big_master":
        gap = v - np.interp(t, tm, vm)
        b.plot(t[m], gap[m], color=c, lw=lw, label=lab)
        b.annotate(f"{e - fm:+.3f}", (t[-1], gap[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                   fontsize=10, color=c)
a.set_ylim(2.6, 4.2); a.set_xlim(2e7, 3.1e8)
a.set_xlabel("training tokens", color=INK2); a.set_ylabel("validation loss (nats per token)", color=INK2)
a.set_title("validation loss; labels: final loss, perplexity", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=10, frameon=False, labelcolor=INK)
b.axhline(0, color="#eb6834", lw=1.4, ls="--")
b.set_ylim(-0.02, 0.45); b.set_xlim(2e7, 3.25e8)
b.set_xlabel("training tokens", color=INK2); b.set_ylabel("loss minus master's at the same tokens", color=INK2)
b.set_title("gap to master weights (labels: final)", color=INK, fontsize=11, loc="left")
b.legend(loc="upper right", fontsize=10, frameon=False, labelcolor=INK)
fig.suptitle("340M ternary model (dim 1024, 24 layers), 300M tokens of Wikipedia: our rule against master weights", color=INK,
             fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/runs_340m.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/runs_340m.png")
