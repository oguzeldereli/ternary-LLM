"""Per-step momentum diagnostics (--lr_diag) along the momentum + look-ahead run: the step-0..345
reproduction (lm_lowrank256_xb2_diag) and the main run from its diagnostic resume on.

  gradient and momentum size         mean |g|, mean |M|, |M| / |g|
  before vs after the proposal       cos(g, g'), g' = cross-batch look-ahead gradient, taken AFTER
                                     the proposed flips (not a clean noise estimate: < 0 = overshoot)
  agreement                          cos(g, M)
  sign agreement                     sign(M) = sign(g), all weights and the top 1% |M|
  subspace                           share of |g| inside M's rank-256 row x column subspace

  python -m scripts.plots.plot_m_diag
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = ["lm_lowrank256_xb2_diag", "lm_lowrank256_xb2_100M"]
TYPES = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]
TCOL = ["#2a78d6", "#00897b", "#7b3fb8", "#111111", "#eb6834", "#c2185b", "#1baf7a"]


def load():
    R = {}
    for d in RUNS:
        for l in open(f"checkpoints/{d}/metrics.jsonl"):
            r = json.loads(l)
            if "d_cos" in r and "d_gg" in r:
                R[r["step"]] = r
    st = np.array(sorted(R))
    return st, (st + 1) * 32768.0, R


def smooth(y, k=10):
    """trailing mean within each contiguous segment (the two runs leave a gap)"""
    y = np.asarray(y, float); out = np.empty_like(y)
    for i in range(len(y)):
        w = min(k, i + 1); out[i] = y[i + 1 - w:i + 1].mean(axis=0)
    return out


def seg_plot(ax, t, y, **kw):
    """plot, breaking the line where the steps jump (no line across the unlogged middle)"""
    br = np.where(np.diff(np.log(t)) > 0.2)[0] + 1
    lab = kw.pop("label", None)
    for j, (a, b) in enumerate(zip(np.r_[0, br], np.r_[br, len(t)])):
        ax.plot(t[a:b], smooth(y[a:b]), label=lab if j == 0 else None, **kw)


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/archive/m_diag.png")
a = ap.parse_args()
st, tok, R = load()
g = lambda k: np.array([R[s][k] for s in st])
gt = lambda k: np.array([R[s][k + "_types"] for s in st])
gabs, mabs, cgm, cgg = g("d_gabs"), g("d_mabs"), g("d_cos"), g("d_gg")

fig, axs = plt.subplots(3, 2, figsize=(20, 18), facecolor=SURFACE)
(a1, a2), (a3, a4), (a5, a6) = axs
seg_plot(a1, tok, gabs, color="#1baf7a", lw=2.2, label="mean |g| (one batch)")
seg_plot(a1, tok, mabs, color="#e53935", lw=2.2, label="mean |M|")
a1.set_yscale("log")
a1b = a1.twinx()
seg_plot(a1b, tok, mabs / gabs, color="#8e8c85", lw=1.6, ls="--", label="|M| / |g| (right axis)")
a1b.axhline(1 / np.sqrt(1 - 0.97 ** 2), color="#8e8c85", lw=0.8, ls=":")
a1b.text(tok[0], 1 / np.sqrt(1 - 0.97 ** 2) * 1.03, "pure-noise level 4.1", color="#8e8c85", fontsize=9)
a1b.axhline(1 / (1 - 0.97), color="#8e8c85", lw=0.8, ls=":")
a1b.text(tok[0], 33 * 1.03, "fully consistent signal 33", color="#8e8c85", fontsize=9)
a1b.set_ylim(0, 40); a1b.tick_params(colors=INK2)
a1b.legend(fontsize=9, frameon=False, loc="lower right")

seg_plot(a2, tok, cgg, color="#111111", lw=2.4,
         label="cos(g, g'): g' = gradient after the proposed flips, on another batch")
seg_plot(a2, tok, cgm, color="#e53935", lw=2.4, label="cos(g, M_{t-1})")
a2.axhline(0, color=INK2, lw=1)

for j, (t_, c) in enumerate(zip(TYPES, TCOL)):
    seg_plot(a3, tok, gt("d_cos")[:, j], color=c, lw=2.0, label=t_)
a3.axhline(0, color=INK2, lw=1)
for j, (t_, c) in enumerate(zip(TYPES, TCOL)):
    seg_plot(a4, tok, gt("d_gg")[:, j], color=c, lw=2.0, label=t_)

seg_plot(a5, tok, g("d_agree"), color="#8e8c85", lw=2.2, label="all weights")
seg_plot(a5, tok, g("d_agree_top"), color="#e53935", lw=2.4, label="top 1% |M| (proposed)")
a5.axhline(0.5, color=INK2, lw=1)

seg_plot(a6, tok, g("d_insub"), color="#111111", lw=2.4, label="all layers")
for j, (t_, c) in enumerate(zip(TYPES, TCOL)):
    seg_plot(a6, tok, gt("d_insub")[:, j], color=c, lw=1.2, label=t_)

titles = ("Size: mean |g| and |M| (left, log); |M| / |g| (right)",
          "Agreement: cos(g, M) and cos(g, g') (g' after the proposed flips: < 0 = overshoot)",
          "cos(g, M_{t-1}) by layer type",
          "cos(g, g') by layer type (gradient before vs after the proposed flips)",
          "Sign agreement sign(M) = sign(g)",
          "Share of |g| inside M's rank-256 subspace")
for ax, tt in zip((a1, a2, a3, a4, a5, a6), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_xscale("log"); ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlim(3e5, 3.1e8)
fig.suptitle("Momentum + look-ahead run, per-step diagnostics (10-step avg); gap = steps not logged",
             x=0.01, ha="left", color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=110, facecolor=SURFACE)
print("wrote", a.out, len(st), "logged steps")
