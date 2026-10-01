"""The 340M runs from the very first step: training loss (trailing mean of the last 20 logged values) and validation loss over the whole run,
log tokens (left) and linear tokens (right).

  python -m scripts.plots.plot_340m_full      -> docs/figures/runs_340m_full.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [("big_master", "master weights", "#eb6834"),
        ("big_dryspend_r1024_s0", "ours, full rank", "#0d1b4c"),
        ("big_dryspend_r512_s0", "ours, rank 512", "#1565c0"),
        ("big_rk128_dryspend_s0", "ours, rank 128", "#64b5f6"),
        ("big_rk64_dryspend_s0", "ours, rank 64", "#9ecae1")]
TPS = 32768


def load(r):
    f = f"checkpoints/{r}_lab/metrics.jsonl"
    if not os.path.exists(f): return None
    tr, va = {}, {}
    for l in open(f):
        x = json.loads(l)
        if "val_loss" in x: va[x["step"]] = x["val_loss"]
        elif "loss" in x: tr[x["step"]] = x["loss"]
    s = np.array(sorted(tr)); l = np.array([tr[k] for k in s])
    k = 20                                     # trailing mean over up to 20 logged points, from the first step on
    cs = np.concatenate([[0.0], np.cumsum(l)]); idx = np.arange(1, len(l) + 1); lo = np.maximum(idx - k, 0)
    sm = (cs[idx] - cs[lo]) / (idx - lo); ss = s
    vs = np.array(sorted(va)); vl = np.array([va[k] for k in vs])
    return (s + 1) * TPS, l, (ss + 1) * TPS, sm, (vs + 1) * TPS, vl


fig, axs = plt.subplots(1, 2, figsize=(22, 8.5), facecolor=SURFACE)
for ax in axs:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
for r, lab, c in RUNS:
    d = load(r)
    if d is None: continue
    t, l, ts, sm, tv, vl = d
    for ax in axs:
        ax.plot(ts, sm, color=c, lw=1.6, alpha=0.9, label=f"{lab}: train (trailing mean of 20 logs)")
        ax.plot(tv, vl, "o", color=c, ms=3.5, label=f"{lab}: validation  [final {vl[-1]:.4f}]")
a = axs[0]
a.set_xscale("log"); a.set_xlim(TPS, 3.1e8); a.set_ylim(2.5, 11)
a.set_xlabel("training tokens (log scale, from the first step)", color=INK2); a.set_ylabel("loss (nats per token)", color=INK2)
a.set_title("whole run, log tokens", color=INK, fontsize=11, loc="left")
a.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK, ncol=1)
b = axs[1]
b.set_xlim(0, 3.1e8); b.set_ylim(2.5, 5.0)
b.set_xlabel("training tokens (linear)", color=INK2); b.set_ylabel("loss (nats per token)", color=INK2)
b.set_title("whole run, linear tokens (y from 2.5 to 5)", color=INK, fontsize=11, loc="left")
fig.suptitle("340M model (d1024_l24), every step: master weights and our rule at rank 64 / 128 / 512 / full", color=INK,
             fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/runs_340m_full.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/runs_340m_full.png")
