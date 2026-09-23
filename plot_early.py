"""Early phase (first ~33M tokens): look-ahead with flip-rate warmup vs look-ahead vs master.

  python3 plot_early.py --out docs/early_phase.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width
    ("p2_baseline",  "master weights (ceiling)",                 "#eb6834", 2.2),
    ("armA_cos_la1", "look-ahead, rate 0.02 cosine",             "#c2185b", 2.6),
    ("la_rwarm",     "look-ahead, rate warmup 0.02 to 0.04",     "#2a78d6", 2.6),
    ("la_r0ramp",    "look-ahead, rate ramp 0 to 0.02",          "#7b3fb8", 2.6),
    ("armA_cosine",  "flips, plain rule (reference)",            "#1baf7a", 1.6),
]
MAX_STEP = 1000


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r for r in rows if "loss" in r and "tokens" in r and r["step"] <= MAX_STEP}
    return tr


def smooth(L, k=50):
    # trailing mean over min(k, ~20% of steps so far): a fixed window drags the
    # starting loss into the first k points and flattens the fast early drop
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/early_phase.png")
a = ap.parse_args()
D = {d: load(d) for d, *_ in RUNS}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)

for d, lab, c, lw in RUNS:
    s = sorted(D[d]); t = np.array([D[d][i]["tokens"] for i in s], float)
    L = smooth(np.array([D[d][i]["loss"] for i in s]))
    ax1.plot(t, L, color=c, lw=lw, label=lab)

# difference vs the look-ahead main run, on identical batches
base = D["armA_cos_la1"]
ax2.axhline(0, color=INK2, lw=1)
for d, lab, c, _ in RUNS[2:4]:
    w = D[d]; s = sorted(set(base) & set(w))
    t = np.array([w[i]["tokens"] for i in s], float)
    ax2.plot(t, smooth(np.array([w[i]["loss"] - base[i]["loss"] for i in s])), color=c, lw=2.4,
             label=lab + " minus look-ahead")
ax2.axvline(305 * 32768, color=GRID, lw=1.5, ls="--")

for d, lab, c, _ in RUNS[1:4]:
    s = sorted(D[d]); t = np.array([D[d][i]["tokens"] for i in s], float)
    ax3.plot(t, smooth(np.array([D[d][i].get("flip_frac", 0) * 100 for i in s]), 20), color=c, lw=2.2,
             label=lab + " (flips kept)")
    ax3.plot(t, [D[d][i]["flip_rate_cfg"] * 100 / 5 for i in s], color=c, lw=1.2, ls="--",
             label=lab + " (rate/5, schedule)")

ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(3e4, 3.5e7); ax1.set_ylim(3.6, 11)
ax1.set_title("Train loss (trailing avg), first 33M tokens (ramp run stops at 10M)", loc="left", color=INK)
ax1.set_ylabel("train loss", color=INK2)
ax2.set_xscale("log"); ax2.set_xlim(3e4, 3.5e7)
ax2.set_title("Loss difference vs look-ahead (same batches); dashed = end of warmup", loc="left", color=INK)
ax2.set_ylabel("Δ train loss (+ = worse than look-ahead)", color=INK2)
ax3.set_xscale("log"); ax3.set_xlim(3e4, 3.5e7)
ax3.set_title("Flips per step (% of weights)", loc="left", color=INK)
ax3.set_ylabel("% of weights", color=INK2)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=130, facecolor=SURFACE)
print("wrote", a.out)
