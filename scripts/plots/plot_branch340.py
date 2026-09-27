"""Branches from ckpt_340 (11M tokens): momentum proposals vs gradient proposals, same
cross-batch look-ahead x2 filter, same batches. Flips, reversals (a flip that undoes the
weight's previous change), net displacement from the branch point, and loss.

  python -m scripts.plots.plot_branch340
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
B = [("branch340_A", "A: momentum M proposes, look-ahead x2 keeps", "#e53935"),
     ("branch340_B", "B: gradient g proposes, look-ahead x2 keeps", "#111111"),
     ("branch340_C", "C: g proposes at rate 0.0072 (flip count of A), look-ahead x2 keeps", "#2a78d6")]
NW = 84_934_656  # ternary weights (sum of N*K over the 84 layers)


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    R = {r["step"]: r for r in rows if "rev_flips" in r}
    st = np.array(sorted(R))
    fl = np.array([R[s]["flip_frac"] for s in st]) * NW
    rv = np.array([R[s]["rev_flips"] for s in st], float)
    net = np.array([R[s]["net_frac"] for s in st]) * NW
    L = np.array([R[s]["loss"] for s in st])
    v = [r["val_loss"] for r in rows if r.get("final")]
    return st, fl, rv, net, L, (v[-1] if v else None)


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/archive/branch340.png")
a = ap.parse_args()
fig, axs = plt.subplots(2, 2, figsize=(18, 12), facecolor=SURFACE)
(a1, a2), (a3, a4) = axs
for d, lab, c in B:
    st, fl, rv, net, L, v = load(d)
    cf = np.cumsum(fl)
    a1.plot(st, cf / 1e6, color=c, lw=2.4, label=f"{lab}: all flips")
    a1.plot(st, net / 1e6, color=c, lw=2.4, ls="--", label=f"{lab}: net changed trits")
    a2.plot(st, np.cumsum(rv) / np.maximum(cf, 1), color=c, lw=2.4, label=lab)
    a3.plot(st, net / np.maximum(cf, 1), color=c, lw=2.4, label=lab)
    k = 10; Ls = np.convolve(L, np.ones(k) / k, mode="valid")
    a4.plot(st[k - 1:], Ls, color=c, lw=2.4, label=lab + (f"  (val {v:.3f})" if v else ""))
    print(f"{d}: flips {cf[-1]/1e6:.2f}M, reversals {np.sum(rv)/cf[-1]*100:.1f}%, net {net[-1]/1e6:.2f}M "
          f"({net[-1]/cf[-1]*100:.0f}% of flips), last-20 train {L[-20:].mean():.3f}, val {v}")
titles = ("Cumulative flips (solid) and net changed trits vs step 340 (dashed), millions",
          "Share of flips that undo the weight's previous change (cumulative)",
          "Net changed trits / all flips (1 = every flip sticks)",
          "Train loss (10-step mean)")
for ax, tt in zip((a1, a2, a3, a4), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("step", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=120, facecolor=SURFACE)
print("wrote", a.out)
