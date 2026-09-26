"""Agreement of the net trit move with the summed gradient vs window length, from several starting points
along training. Ours: 10-step continuations from the momentum run's checkpoints (1, 2, 5, 10) plus the
100-step windows of the replay; master: 100-step windows of master_tracked (and short windows once measured).

  python -m scripts.plots.plot_window_curves
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STARTS = [340, 1000, 2000, 3000, 4000, 5000, 6000, 6243]
cm = plt.get_cmap("viridis")


def win100(d, s):
    W = {r["step"]: r for r in map(json.loads, open(f"checkpoints/{d}/metrics.jsonl")) if "w_steps" in r}
    ks = [k for k in W if s < k <= s + 150]
    return W[min(ks)] if ks else None


def curve(d):
    R = {}
    for r in map(json.loads, open(f"checkpoints/{d}/metrics.jsonl")):
        R.update({k: v for k, v in r.items() if k.startswith("c") and isinstance(v, float)})
    return R


fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=SURFACE)
for j, s in enumerate(STARTS):
    if not os.path.exists(f"checkpoints/curve_ours_{s}/metrics.jsonl"): continue
    R = curve(f"curve_ours_{s}"); w = win100("r4090_replay_11M_205M", s)
    Ws = [1, 2, 5, 10] + ([100] if w else [])
    col = cm(j / (len(STARTS) - 1)); lab = f"ours from {(s + 1) * 32768 / 1e6:.0f}M"
    a1.plot(Ws, [R[f"c{W}_agree_S"] for W in (1, 2, 5, 10)] + ([w["w_agree_S"]] if w else []), "o-", color=col, lw=2, label=lab)
    a2.plot(Ws, [R[f"c{W}_cos_S"] for W in (1, 2, 5, 10)] + ([w["w_cos_S"]] if w else []), "o-", color=col, lw=2, label=lab)
    m = win100("master_tracked", s)
    if m:
        a1.plot([100], [m["w_agree_S"]], "s", color=col, ms=9, mec="#eb6834", mew=2)
        a2.plot([100], [m["w_cos_S"]], "s", color=col, ms=9, mec="#eb6834", mew=2)
if os.path.exists("checkpoints/curve_master/metrics.jsonl"):
    pass  # master's short windows, once measured (scripts/remote/master_curve.sh)
a1.axhline(0.5, color=INK2, lw=1)
a1.plot([], [], "s", color="white", mec="#eb6834", mew=2, ms=9, label="master, 100-step window (same start)")
a1.set_title("Net trit moves along -(summed gradient of the window), vs window length", loc="left", color=INK)
a2.set_title("cos(net trit move, -summed gradient), vs window length", loc="left", color=INK)
for ax in (a1, a2):
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("window length (steps)", color=INK2); ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, lw=0.6); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
a1.set_ylim(0.45, 1.0)
fig.tight_layout()
fig.savefig("docs/figures/window_curves.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/window_curves.png")
