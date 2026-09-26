"""Momentum switched off at 11M tokens (step 340): look-ahead x2 alone from the momentum run's
checkpoint (same rate / flip count matched at 11M), against the momentum run (original and the 4090
replay), look-ahead alone from scratch, and master. Same batches from step 340 on.

  python -m scripts.plots.plot_momentum_off
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width, style
    ("p2_baseline", "master weights + AdamW", "#eb6834", 2.2, "-"),
    ("lm_lowrank256_xb2_100M", "momentum + look-ahead x2 (original, laptop)", "#e53935", 2.8, "-"),
    ("r4090_replay_11M_205M", "momentum + look-ahead x2 (replay from 11M, 4090)", "#b71c1c", 1.6, "--"),
    ("r4090_la_from11M", "momentum OFF at 11M: look-ahead x2, same rate", "#2a78d6", 2.6, "-"),
    ("r4090_la_from11M_matched", "momentum OFF at 11M: look-ahead x2, rate x0.36", "#7b3fb8", 2.6, "-"),
    ("overnight_full", "look-ahead x2 from scratch (no momentum ever)", "#111111", 1.8, "-"),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r}
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    k = 50; c = np.cumsum(np.insert(L, 0, 0.0))
    Ls = np.array([(c[i + 1] - c[max(0, i + 1 - k)]) / min(k, i + 1) for i in range(len(L))])
    v = sorted((r["step"], r["val_loss"]) for r in rows if "val_loss" in r)
    return (s + 1) * 32768.0, Ls, np.array([(a + 1) * 32768.0 for a, _ in v]), np.array([b for _, b in v])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/momentum_off.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor=SURFACE)
for d, lab, c, lw, ls in RUNS:
    if not os.path.exists(f"checkpoints/{d}/metrics.jsonl"): continue
    t, L, vt, vl = load(d)
    ax1.plot(t, L, color=c, lw=lw, ls=ls, label=lab)
    if len(vl): ax2.plot(vt, np.exp(vl), "o" + ls, color=c, lw=lw, ms=4, label=lab)
for ax in (ax1, ax2):
    ax.axvline(341 * 32768, color=INK2, lw=1, ls=":")
    ax.text(341 * 32768 * 1.03, 0.97, "momentum off\n(step 340, 11M)", transform=ax.get_xaxis_transform(),
            va="top", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e6, 1.2e8)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
ax1.set_ylim(3.2, 6.0); ax2.set_ylim(22, 250)
ax1.set_title("Train loss (50-step mean)", loc="left", color=INK, fontsize=12)
ax2.set_title("Validation perplexity", loc="left", color=INK, fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=120, facecolor=SURFACE)
print("wrote", a.out)
