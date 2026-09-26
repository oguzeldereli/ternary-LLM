"""Overnight 4090 results: (a) 300-step branches from 205M (ckpt_6243): momentum horizon beta
0.97 / 0.99 / 0.995 and gradient proposals at matched flip counts; (b) 10M screens: stateless
look-ahead at rate 0.007 / 0.013 / 0.02 against momentum + look-ahead.

  python -m scripts.plots.plot_overnight_branches
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r}
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    v = [r["val_loss"] for r in rows if r.get("final")]
    return s, L, (v[-1] if v else None)


def sm(L, k):
    c = np.cumsum(np.insert(L, 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - k)]) / min(k, i + 1) for i in range(len(L))])


fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7), facecolor=SURFACE)
for d, lab, c, lw in (("b6243_A", "M proposes, beta 0.97", "#d62728", 2.4),
                      ("b6243_A_beta0.99", "M proposes, beta 0.99", "#7b3fb8", 2.4),
                      ("b6243_A_beta0.995", "M proposes, beta 0.995", "#2a78d6", 2.0),
                      ("b6243_C2", "g proposes, flip count of M", "#1baf7a", 2.0),
                      ("b6243_C1", "g proposes, 0.5 x rate", "#8e8c85", 1.6),
                      ("b6243_B", "g proposes, same rate", "#111111", 1.6)):
    s, L, v = load(d)
    a1.plot(s, sm(L, 50), color=c, lw=lw, label=f"{lab}  (val {v:.4f})")
a1.set_title("(a) 300 steps from 205M tokens, same batches: train loss (50-step mean)", loc="left", color=INK)
a1.set_xlabel("step", color=INK2)
for d, lab, c, lw in (("lm_lowrank256_xb2", "momentum + look-ahead x2, r 0.02 (laptop)", "#d62728", 2.6),
                      ("la_xb2", "look-ahead x2, r 0.02 (laptop)", "#111111", 2.0),
                      ("r4090_la_xb2_r013", "look-ahead x2, r 0.013 (4090)", "#2a78d6", 2.2),
                      ("r4090_la_xb2_r007", "look-ahead x2, r 0.007 (4090)", "#7b3fb8", 2.0)):
    s, L, v = load(d)
    a2.plot((s + 1) * 32768, sm(L, 15), color=c, lw=lw, label=f"{lab}  (val {v:.3f})")
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1e6, 1.1e7); a2.set_ylim(4.7, 9)
a2.set_title("(b) 10M-token screens: stateless look-ahead at three rates vs momentum (15-step mean)",
             loc="left", color=INK)
a2.set_xlabel("tokens", color=INK2)
for ax in (a1, a2):
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9.5, frameon=False)
fig.tight_layout()
fig.savefig("docs/figures/overnight_branches.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/overnight_branches.png")
