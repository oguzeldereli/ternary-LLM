"""Net trit moves of the first k steps of a 100-step window, scored against the summed gradient of the
whole window (left) and of its later part only, excluding the batches that chose the moves (right).
Ours: 100-step continuations from the momentum run's checkpoints; master: windows opened at the same
steps in a master run with the same batches.

  python -m scripts.plots.plot_window_curves
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
KS = [1, 2, 5, 10, 20, 50]


def get(d):
    out = {}
    if os.path.exists(f"checkpoints/{d}/metrics.jsonl"):
        for r in map(json.loads, open(f"checkpoints/{d}/metrics.jsonl")):
            if "c100_d1_agree" in r:
                out[r["c_start"]] = r
    return out


M = get("curve_master"); O = {}
for s in (340, 2000, 4000, 6000):
    O.update(get(f"curve100_ours_{s}"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7.5), facecolor=SURFACE)
cm = plt.get_cmap("viridis")
starts = [340, 2000, 4000, 6000]
for j, s in enumerate(starts):
    col = cm(j / (len(starts) - 1)); tok = f"{(s + 1) * 32768 / 1e6:.0f}M"
    for D, ls, mk, who in ((M, "-", "s", "master"), (O, "--", "o", "ours")):
        if s not in D: continue
        r = D[s]
        a1.plot(KS + [100], [r[f"c100_d{k}_agree"] for k in KS] + [r["c100_agree_S"]], ls, marker=mk, color=col,
                lw=2.2, ms=6, label=f"{who} from {tok}")
        a2.plot(KS, [r[f"c100_d{k}_agree_later"] for k in KS], ls, marker=mk, color=col, lw=2.2, ms=6,
                label=f"{who} from {tok}")
a1.set_title("Moves of the first k steps vs the summed gradient of the whole 100-step window\n"
             "(solid squares: master; dashed circles: ours; k = 100 is the window's net move)", loc="left", color=INK)
a2.set_title("Moves of the first k steps vs the summed gradient of steps k+1..100 only\n"
             "(the batches that chose the moves excluded)", loc="left", color=INK)
for ax in (a1, a2):
    ax.axhline(0.5, color=INK2, lw=1)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("k (steps)", color=INK2); ax.set_ylabel("share of moved trits along -gradient", color=INK2)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=8.5, frameon=False, ncol=2)
a1.set_ylim(0.35, 1.0); a2.set_ylim(0.35, 0.6)
fig.tight_layout()
fig.savefig("docs/figures/window_curves.png", dpi=110, facecolor=SURFACE)
print("wrote docs/figures/window_curves.png")
