"""From-scratch 20M comparison of the momentum fixes (momentum + look-ahead x2 settings, same batches):
train loss, validation, and train loss minus the mean of the baseline seeds. Plots whatever has run so far.

  python -m scripts.plots.plot_s20
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width, style
    ("fp32_baseline", "full precision fp32 + AdamW (reference)", "#00838f", 2.0, ":"),
    ("master_tracked", "master weights + AdamW (reference)", "#eb6834", 2.0, ":"),
    ("lm_lowrank256_xb2_100M", "momentum + look-ahead x2, original 300M run (laptop)", "#6d4c41", 2.0, ":"),
    ("s20_base", "baseline (seed 0)", "#111111", 2.4, "-"),
    ("s20_base_seed1", "baseline (seed 1)", "#555555", 1.6, "-"),
    ("s20_base_seed2", "baseline (seed 2)", "#999999", 1.6, "-"),
    ("s20_gate", "sign gate", "#d62728", 2.8, "-"),
    ("s20_spend2", "spend (c=2)", "#1baf7a", 1.8, "-"),
    ("s20_refresh16", "subspace refresh", "#2a78d6", 1.8, "-"),
    ("s20_vnorm99", "vnorm (factored 2nd moment)", "#7b3fb8", 1.8, "-"),
    ("s20_maskstuck", "stuck mask", "#eb6834", 1.8, "-"),
    ("s20_magadd4", "low-rank magnitude, additive r4", "#00acc1", 1.8, "--"),
    ("s20_magmul4", "low-rank magnitude, multiplicative r4", "#a1887f", 1.8, "--"),
    ("s20_gate_maskstuck", "gate + stuck mask", "#e91e63", 2.2, "--"),
    ("s20_gate_vnorm99", "gate + vnorm", "#ff9800", 2.2, "--"),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r and r["step"] <= 611}
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    k = 30; c = np.cumsum(np.insert(L, 0, 0.0))
    Ls = np.array([(c[i + 1] - c[max(0, i + 1 - min(k, i // 3 + 1))]) / min(k, i // 3 + 1) for i in range(len(L))])
    v = sorted({r["step"]: r["val_loss"] for r in rows if "val_loss" in r and r["step"] <= 611}.items())
    return s, Ls, v


D = {d: load(d) for d, *_ in RUNS if os.path.exists(f"checkpoints/{d}/metrics.jsonl")}
base = [D[d] for d in ("s20_base", "s20_base_seed1", "s20_base_seed2") if d in D]
n = min(len(b[0]) for b in base)
bmean = np.mean([b[1][:n] for b in base], axis=0); bsteps = base[0][0][:n]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(24, 7), facecolor=SURFACE)
for d, lab, c, lw, ls in RUNS:
    if d not in D: continue
    s, L, v = D[d]
    fin = [x for st, x in v if st >= 610]
    lab2 = lab + (f"  (val {fin[-1]:.4f})" if fin else "")
    a1.plot((s + 1) * 32768, L, color=c, lw=lw, ls=ls, label=lab2)
    if v: a2.plot([(st + 1) * 32768 for st, _ in v], [np.exp(x) for _, x in v], "o" + ls, color=c, lw=lw, ms=5, label=lab)
    m = min(len(s), n)
    a3.plot((s[:m] + 1) * 32768, L[:m] - bmean[:m], color=c, lw=lw, ls=ls, label=lab)
if len(base) > 1:
    lo = np.min([b[1][:n] for b in base], 0) - bmean; hi = np.max([b[1][:n] for b in base], 0) - bmean
    a3.fill_between((bsteps + 1) * 32768, lo, hi, color="#999999", alpha=0.25, lw=0, label="baseline seed range")
a3.axhline(0, color=INK2, lw=1)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(3e4, 2.1e7); a1.set_ylim(4.2, 11)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(5e6, 2.2e7)
a3.set_xscale("log"); a3.set_xlim(3e4, 2.1e7); a3.set_ylim(-0.25, 0.25)
titles = ("Train loss (30-step mean), log-log", "Validation perplexity, log-log",
          "Train loss minus the baseline-seed mean (< 0 = better)")
for ax, tt in zip((a1, a2, a3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=8.5, frameon=False)
fig.suptitle("Momentum fixes from scratch to 20M tokens (momentum + cross-batch look-ahead x2, same batches)",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("docs/figures/s20.png", dpi=105, facecolor=SURFACE)
print("wrote docs/figures/s20.png", sorted(D))
