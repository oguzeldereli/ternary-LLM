"""fp32 float tail vs the old bf16 tail (norm gains frozen at 1.0) vs master, first ~10M tokens.

  python3 plot_tail.py --out docs/tail_fp32.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [("p2_baseline", "master weights", "#eb6834"),
        ("la_fastramp_lr15", "flips, bf16 tail (norm gains frozen)", "#8e8c85"),
        ("la_fast_fp32tail", "flips, fp32 tail (norm gains train)", "#c2185b")]
MAX_STEP = 320


def load(d):
    tr, pr = {}, []
    for l in open(f"checkpoints/{d}/metrics.jsonl"):
        r = json.loads(l)
        if r.get("probe"): pr.append(r)
        elif "loss" in r and "tokens" in r and r["step"] <= MAX_STEP: tr[r["step"]] = r
    return tr, pr


def smooth(L, k=25):
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/tail_fp32.png")
a = ap.parse_args()
D = {d: load(d) for d, _, _ in RUNS}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)
base = D["p2_baseline"][0]
for d, lab, c in RUNS:
    tr, pr = D[d]; s = sorted(tr)
    t = np.array([tr[i]["tokens"] for i in s], float)
    ax1.plot(t, smooth(np.array([tr[i]["loss"] for i in s])), color=c, lw=2.4, label=lab)
    if d != "p2_baseline":
        s2 = [i for i in s if i in base]
        ax2.plot([tr[i]["tokens"] for i in s2], smooth(np.array([tr[i]["loss"] - base[i]["loss"] for i in s2])),
                 color=c, lw=2.4, label=lab + " minus master")
for d, lab, c in (("probe_master", "master weights", "#eb6834"),
                  ("la_fast_fp32tail", "flips, fp32 tail", "#c2185b"),
                  ("probe_fast", "flips, bf16 tail", "#8e8c85")):
    if not os.path.exists(f"checkpoints/{d}/metrics.jsonl"): continue
    pr = load(d)[1]
    ax3.plot([(r["step"] + 1) * 32768 for r in pr], [r["norm_gain"] for r in pr], color=c, lw=2.4, label=lab)
ax2.axhline(0, color=INK2, lw=1)
titles = ("Train loss (trailing avg)", "Train loss minus master (same batches)", "Mean |final-norm gain|")
for ax, tt in zip((ax1, ax2, ax3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_xscale("log"); ax.set_xlim(3e4, 1.1e7)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
ax1.set_yscale("log"); ax1.set_ylim(5, 11)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
