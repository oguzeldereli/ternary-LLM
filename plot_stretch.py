"""Is look-ahead a vertically squashed copy of master?

Look-ahead is cut at the plateau (T0 = 10^5.5 tokens), its tokens re-counted from the
cut so it starts at master's first step, and its log-loss drop is scaled by one factor s
about its start, with the start moved onto master's start:
    log L'(x) = log Lm(x0) + s * (log L(t) - log L(T0)),   x = t - T0 + one step
s is chosen so the last look-ahead point lands exactly on master at the same x.

  python3 plot_stretch.py --out docs/stretch.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
T0, STEP_TOK = 10 ** 5.5, 32768


def load(d):
    tr = {}
    for l in open(f"checkpoints/{d}/metrics.jsonl"):
        r = json.loads(l)
        if "loss" in r and "tokens" in r: tr[r["step"]] = r
    s = sorted(tr)
    t = np.array([tr[i]["tokens"] for i in s], float)
    L = np.array([tr[i]["loss"] for i in s])
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(50, i // 5 + 1) for i in range(len(L))]
    return t, np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/stretch.png")
a = ap.parse_args()
mt, mL = load("p2_baseline")
lt, lL = load("armA_cos_la1")
k = lt >= T0
x = lt[k] - lt[k][0] + STEP_TOK
y = lL[k]
m_at = lambda xx: np.exp(np.interp(np.log(xx), np.log(mt), np.log(mL)))
y0, m0 = np.log(y[0]), np.log(mL[0])
s = (np.log(m_at(x[-1])) - m0) / (np.log(y[-1]) - y0)
ys = np.exp(m0 + s * (np.log(y) - y0))
ratio = ys / m_at(x)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), facecolor=SURFACE,
                               gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
ax1.plot(mt, mL, color="#eb6834", lw=2.6, label="master weights (baseline)")
ax1.plot(x, y, color="#c2185b", lw=1.2, alpha=0.45, label="look-ahead, cut at 10^5.5 and re-based (unstretched)")
ax1.plot(x, ys, color="#c2185b", lw=2.6, ls="--",
         label=f"look-ahead, start moved onto master's, log-drop stretched ×{s:.2f}")
ax1.set_yscale("log"); ax1.set_ylim(2.5, 11)
ax1.set_ylabel("train loss", color=INK2)
ax1.set_title(f"Look-ahead stretched vertically by {s:.2f}× (in log-loss) to end on master",
              color=INK, fontsize=12, loc="left")
ax2.axhline(1, color=INK2, lw=1)
ax2.plot(x, ratio, color="#c2185b", lw=2)
ax2.set_ylabel("stretched / master", color=INK2)
ax2.set_xlabel("tokens (look-ahead counted from the cut)", color=INK2)
for ax in (ax1, ax2):
    ax.set_xscale("log"); ax.set_xlim(3e4, 3.5e8)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
ax1.legend(fontsize=9.5, frameon=False, loc="lower left")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
q = x >= 1e6
print(f"wrote {a.out}  stretch s={s:.3f}  ratio past 1M tokens: min {ratio[q].min():.3f} max {ratio[q].max():.3f} "
      f"median |dev| {np.median(np.abs(ratio[q]-1))*100:.1f}%")
