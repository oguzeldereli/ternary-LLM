"""Where do the trits move, ours vs master: 100-step window measurements (--move_window) and per-step
flip tracking. Ours = momentum + look-ahead x2 (replay 11M -> 205M on the 4090, plus the original
run's per-step logs); master = latent weights + AdamW (master_tracked, same batches).

  python -m scripts.plots.plot_windows
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
OURS, MASTER = "#d62728", "#eb6834"
TOK = 32768.0


def rows(d):
    return [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]


def windows(d):
    R = sorted((r["step"], r) for r in rows(d) if "w_steps" in r)
    return np.array([(s + 1) * TOK for s, _ in R]), [r for _, r in R]


def smooth(y, k):
    """trailing mean over k points, ignoring NaN (first windows have no previous window / momentum)"""
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        w = y[max(0, i + 1 - k):i + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/windows.png")
a = ap.parse_args()
to, Wo = windows("r4090_replay_11M_205M")
tm, Wm = windows("master_tracked")
g = lambda W, k: np.array([w.get(k, np.nan) for w in W])
fig, axs = plt.subplots(3, 3, figsize=(24, 18), facecolor=SURFACE)
axs = axs.ravel()
panels = [
    ("w_agree_S", "Net trit moves along -(summed gradient of the window)\nshare of moved trits (0.5 = chance)", (0.4, 1.02)),
    ("w_cos_S", "cos(net trit move, -summed gradient)\n(master's latent-weight move: dashed)", (-0.05, 0.9)),
    ("w_moved", "Share of trits changed per 100-step window (net)", None),
    ("w_agree_M", "Net trit moves along -momentum at the window start\n(ours: M; master: Adam's first moment)", (0.4, 1.02)),
    ("w_cos_SM", "cos(summed gradient of the window, momentum at its start)", (-0.3, 0.3)),
    ("w_coh", "Gradient coherence |sum g|^2 / sum |g|^2 over the window\n(1 = independent noise, 100 = one direction, <1 = anti-correlated)", None),
    ("w_cos_Sprev", "cos(summed gradient, previous window's summed gradient)", (-0.3, 0.3)),
]
for ax, (k, title, yl) in zip(axs, panels):
    ax.plot(tm, smooth(g(Wm, k), 3), color=MASTER, lw=2.4, label="master weights + AdamW")
    ax.plot(to, smooth(g(Wo, k), 3), color=OURS, lw=2.4, label="ours: momentum + look-ahead x2")
    if k == "w_cos_S":
        ax.plot(tm, smooth(g(Wm, "w_cos_lat_S"), 3), color=MASTER, lw=1.6, ls="--",
                label="master latent weights (float move)")
    if k in ("w_agree_S", "w_agree_M"):
        ax.axhline(0.5, color=INK2, lw=1)
    if k in ("w_cos_SM", "w_cos_Sprev"):
        ax.axhline(0, color=INK2, lw=1)
    if k == "w_coh":
        ax.axhline(1, color=INK2, lw=1)
    if yl: ax.set_ylim(*yl)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
# per-step flips and reversals
NW = 84_934_656
def perstep(d):
    R = sorted((r["step"], r) for r in rows(d) if "flip_frac" in r)
    return np.array([(s + 1) * TOK for s, _ in R]), [r for _, r in R]
ax = axs[7]
for d, c, lab, lw in (("master_tracked", MASTER, "master", 2.4),
                      ("lm_lowrank256_xb2_diag", OURS, "ours (0-11M)", 2.4),
                      ("r4090_replay_11M_205M", OURS, "ours (11-205M, replay)", 2.4),
                      ("lm_lowrank256_xb2_100M", OURS, None, 1.2)):
    t, R = perstep(d)
    m = t > (6244 * TOK if d == "lm_lowrank256_xb2_100M" else 0)
    ax.plot(t[m], smooth([r["flip_frac"] * 100 for r in R], 50)[m], color=c, lw=lw, label=lab)
ax.set_yscale("log"); ax.set_title("Trits changed per step (%)", loc="left", color=INK, fontsize=11)
ax = axs[8]
for d, c, lab in (("master_tracked", MASTER, "master (tracked from step 0)"),
                  ("r4090_replay_11M_205M", OURS, "ours (tracked from 11M)")):
    t, R = perstep(d)
    fl = np.array([r["flip_frac"] * NW for r in R]); rv = np.array([r.get("rev_flips", 0) for r in R], float)
    ax.plot(t, smooth(rv, 100) / np.maximum(smooth(fl, 100), 1), color=c, lw=2.4, label=lab + ": reversals")
    ax.plot(t, [r["never_frac"] for r in R], color=c, lw=1.6, ls="--", label=lab + ": never changed")
ax.set_ylim(0, 1.02)
ax.set_title("Flips that undo the weight's previous change (solid)\nshare of trits never changed (dashed)",
             loc="left", color=INK, fontsize=11)
for ax in axs:
    ax.set_xscale("log"); ax.set_xlim(1e6, 3.2e8); ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, lw=0.6); ax.set_xlabel("tokens", color=INK2)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.legend(fontsize=9, frameon=False)
fig.suptitle("Where the trits move: ours (momentum + look-ahead) vs master weights, same batches "
             "(window = 100 steps = 3.3M tokens; 3-window mean)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=100, facecolor=SURFACE)
print("wrote", a.out)
