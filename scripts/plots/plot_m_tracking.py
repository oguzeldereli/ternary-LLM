"""How well does the low-rank momentum M track the gradient along the long run?
cos(g_t, M_{t-1}) per ternary layer (logged every step), by layer type and depth; and the
size of the per-weight gradient along training (only logged by the plain-flip runs).

  python -m scripts.plots.plot_m_tracking
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUN = "lm_lowrank256_xb2_100M"
TYPES = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]      # module order in a block
TCOL = ["#2a78d6", "#00897b", "#7b3fb8", "#111111", "#eb6834", "#c2185b", "#1baf7a"]


def smooth(y, k):
    c = np.cumsum(np.insert(y, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(y)
    for i in range(len(y)):
        w = min(k, i + 1); out[i] = (c[i + 1] - c[i + 1 - w]) / w
    return out


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/m_tracking.png")
a = ap.parse_args()
rows = [json.loads(l) for l in open(f"checkpoints/{RUN}/metrics.jsonl")]
R = {r["step"]: r for r in rows if "lr_cos_layers" in r}
st = np.array(sorted(R)); tok = (st + 1) * 32768.0
C = np.array([R[s]["lr_cos_layers"] for s in st])                 # [steps, 84]
fig, axs = plt.subplots(2, 2, figsize=(20, 12), facecolor=SURFACE)
(ax1, ax2), (ax3, ax4) = axs
Cs = smooth(C, 100)
ax1.fill_between(tok, np.percentile(Cs, 10, axis=1), np.percentile(Cs, 90, axis=1), color="#e53935",
                 alpha=0.18, lw=0, label="10-90% of the 84 layers")
ax1.plot(tok, smooth(C.mean(1), 20), color="#e53935", lw=0.8, alpha=0.5, label="mean, 20-step avg")
ax1.plot(tok, Cs.mean(1), color="#b71c1c", lw=2.6, label="mean, 100-step avg")
ax1.axhline(0, color=INK2, lw=1)
for j, (t, c) in enumerate(zip(TYPES, TCOL)):
    ax2.plot(tok, smooth(C[:, j::7].mean(1), 100), color=c, lw=2.2, label=t)
ax2.axhline(0, color=INK2, lw=1)
nb = C.shape[1] // 7
cm = plt.get_cmap("viridis")
for b in range(nb):
    ax3.plot(tok, smooth(C[:, b * 7:(b + 1) * 7].mean(1), 100), color=cm(b / max(nb - 1, 1)), lw=1.8,
             label=f"block {b}" if b in (0, nb // 2, nb - 1) else None)
ax3.axhline(0, color=INK2, lw=1)
# gradient size along training: the plain-flip runs log mean|g| per layer (this run does not)
for d, lab, col in (("armA_cosine", "plain flips, 300M schedule", "#1baf7a"),
                    ("armA_cos_600M", "plain flips, 600M schedule", "#8e8c85")):
    rr = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    g = [(r["tokens"], np.mean(r["gmean_layers"])) for r in rr if "gmean_layers" in r and "tokens" in r]
    t, gm = np.array(g).T
    ax4.plot(t, smooth(gm, 100), color=col, lw=2.2, label=lab)
ax4.set_yscale("log")
titles = ("cos(g_t, M_{t-1}): current gradient vs rank-256 momentum (mean over layers)",
          "cos(g, M) by layer type (mean over the 12 blocks, 100-step avg)",
          "cos(g, M) by depth (mean over the 7 layers of a block)",
          "Gradient size along training: mean |dL/dT| per weight (plain-flip runs)")
for ax, tt in zip((ax1, ax2, ax3, ax4), titles):
    ax.set_title(tt, loc="left", color=INK, fontsize=11.5)
    ax.set_xscale("log"); ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
for ax in (ax1, ax2, ax3):
    ax.set_xlim(1e6, 3.1e8); ax.set_ylim(-0.3, 0.5)
ax4.set_xlim(1e6, 6.1e8)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
