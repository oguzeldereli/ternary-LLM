"""Ternary MLP testbed: master vs stateless flips vs look-ahead (val loss, log-log),
the local slope, and the transformer-style stretch test (flip curve's log-loss drop
scaled onto master's).

  python -m scripts.plots.plot_mlp
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [("mlp_master", "master weights (BitNet STE + AdamW)", "#eb6834"),
        ("mlp_flip", "stateless flips", "#1baf7a"),
        ("mlp_xb2", "flips + cross-batch look-ahead x2", "#111111")]


def load(n):
    f = f"checkpoints/mlp/{n}.jsonl"
    if not os.path.exists(f): return None
    v = [json.loads(l) for l in open(f)]
    v = [r for r in v if "val_loss" in r]
    if len(v) < 6: return None
    return np.array([r["examples"] for r in v], float), np.array([r["val_loss"] for r in v])


def slope(t, L, lo, hi):
    m = (t >= lo) & (t <= hi)
    return np.polyfit(np.log(t[m]), np.log(L[m]), 1)[0] if m.sum() > 2 else np.nan


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/mlp.png")
a = ap.parse_args()
D = {n: load(n) for n, *_ in RUNS}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), facecolor=SURFACE)
for n, lab, c in RUNS:
    if D[n] is None: continue
    t, L = D[n]
    ax1.plot(t, L, color=c, lw=2.4, label=lab)
    mid = np.sqrt(t[1:] * t[:-1])
    ls = np.diff(np.log(L)) / np.diff(np.log(t))
    k = 5
    ax2.plot(mid[k - 1:], np.convolve(ls, np.ones(k) / k, "valid"), color=c, lw=2.2, label=lab)
M = D["mlp_master"]
if M is not None:
    tm, Lm = M
    ax3.plot(tm, Lm, color="#eb6834", lw=2.6, label="master weights")
    for n, lab, c in RUNS[1:]:
        if D[n] is None: continue
        t, L = D[n]
        cut = np.searchsorted(t, t[0] * 3)               # skip the first drop, as for the LM
        t, L = t[cut:], L[cut:]
        x = t - t[0] + tm[0]
        m_at = lambda xx: np.exp(np.interp(np.log(xx), np.log(tm), np.log(Lm)))
        s = (np.log(m_at(x[-1])) - np.log(Lm[0])) / (np.log(L[-1]) - np.log(L[0]))
        ys = np.exp(np.log(Lm[0]) + s * (np.log(L) - np.log(L[0])))
        ax3.plot(x, ys, color=c, lw=2.2, ls="--", label=f"{lab}, log-drop x{s:.2f}")
        print(f"{n}: stretch {s:.2f}, max |stretched/master - 1| past 10% of run: "
              f"{np.max(np.abs(ys[len(ys)//10:] / m_at(x[len(x)//10:]) - 1)) * 100:.1f}%")
for n, *_ in RUNS:
    if D[n] is not None:
        t, L = D[n]; T = t[-1]
        print(f"{n}: final val {L[-1]:.3f}; slope {slope(t, L, T/100, T/10):+.3f} (1-10%), "
              f"{slope(t, L, T/10, T):+.3f} (10-100%)")
titles = ("Val loss", "Local slope d log L / d log examples", "Stretch test: flips' log-drop scaled to end on master")
for ax, tt in zip((ax1, ax2, ax3), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_xscale("log"); ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("training examples", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
ax1.set_yscale("log"); ax3.set_yscale("log")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
