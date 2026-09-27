"""The running long run against the reference runs; safe to re-run while it trains.

  python -m scripts.plots.plot_live
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
RUNS = [  # dir, label, color, width, highlight
    ("fp32_baseline", "full precision fp32 + AdamW (not ternary)", "#00acc1", 2.2, False),
    ("p2_baseline",   "master weights", "#eb6834", 2.2, False),
    ("lm_lowrank256_xb2_100M", "momentum + look-ahead (baseline)", "#e53935", 2.0, False),
    ("magadd_full", "momentum + look-ahead + additive r4", "#1565c0", 2.0, False),
    ("nola_lab", "MOMENTUM, NO LOOK-AHEAD (induction from ~98M)", "#7b1fa2", 3.0, True),
    ("nola_then_la", "NO LOOK-AHEAD, THEN LOOK-AHEAD FROM 131M", "#00897b", 3.0, True),
    ("la_sched", "LOOK-AHEAD 0-30M, OFF 30-131M, ON AFTER", "#fbc02d", 3.0, True),
    ("la_sched98", "LOOK-AHEAD 0-30M, OFF 30-98M, ON AFTER", "#6d4c41", 3.0, True),
    ("nola_add16", "no look-ahead + additive r16 (stopped: adapter takes over q/k)", "#c2185b", 1.6, False),
]


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = [r for r in rows if "loss" in r and "tokens" in r]
    t = np.array([r["tokens"] for r in tr], float); L = np.array([r["loss"] for r in tr])
    tok_per_step = t[1] - t[0]
    v = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    vt = np.array([(s + 1) * tok_per_step for s, _ in v], float)
    return t, L, vt, np.array([np.exp(x) for _, x in v])


def smooth(L, k=100):
    # trailing mean over min(k, ~20% of steps so far): a fixed window drags the
    # starting loss into the first k points and flattens the fast early drop
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/live.png")
a = ap.parse_args()
fig, axs = plt.subplots(2, 2, figsize=(20, 13), facecolor=SURFACE)
(ax1, ax2), (ax3, ax4) = axs
G = np.logspace(np.log10(1e6), np.log10(6e8), 120)
tm_, Lm_, *_ = load("p2_baseline"); Lm_ = smooth(Lm_)
lnm = lambda t: np.interp(np.log(t), np.log(tm_), np.log(Lm_))
CS = np.linspace(0.5, 4.0, 351)
for d, lab, c, lw, hi in RUNS:
    if not os.path.exists(f"checkpoints/{d}/metrics.jsonl"): continue
    t, L, vt, vp = load(d)
    Ls = smooth(L)
    z = 6 if hi else 4
    ax1.plot(t, Ls, color=c, lw=lw, label=lab, zorder=z)
    if len(vp):
        ax2.plot(vt, vp, "o-", color=c, lw=lw, ms=5, label=lab, zorder=z)
        if hi:
            ax2.annotate(f"{vp[-1]:.1f}", (vt[-1], vp[-1]), textcoords="offset points",
                         xytext=(8, 4), color=c, fontsize=11, weight="bold")
    # local power-law exponent d ln L / d ln t, over a factor-1.6 window of tokens
    g = G[(G > t[0] * 1.3) & (G < t[-1] / 1.3)]
    if len(g) > 2:
        lo = np.interp(np.log(g / 1.25), np.log(t), np.log(Ls))
        hi_ = np.interp(np.log(g * 1.25), np.log(t), np.log(Ls))
        ax3.plot(g, (hi_ - lo) / np.log(1.25 ** 2), color=c, lw=lw, label=lab, zorder=z)
    # token stretch vs master: in a factor-2 window around each point, the c for which
    # L_run(t) ~ L_master(t / c) fits best (c > 1: the run needs c x master's tokens)
    if d != "p2_baseline":
        g2 = G[(G > t[0] * 1.5) & (G < t[-1] / 1.5) & (G > 2e6)]
        cc, ee = [], []
        for x in g2:
            w = np.logspace(np.log10(x / 1.41), np.log10(x * 1.41), 12)
            y = np.interp(np.log(w), np.log(t), np.log(Ls))
            err = [np.mean(np.abs(y - lnm(w / k))) for k in CS if (w / k).max() <= tm_[-1]]
            if err:
                i = int(np.argmin(err)); cc.append(CS[i]); ee.append(err[i])
        if cc:
            ax4.plot(g2[:len(cc)], cc, color=c, lw=lw, label=lab, zorder=z)
for ax, yl, title in ((ax1, "train loss (100-step avg)", "Train loss, log-log"),
                      (ax2, "val perplexity", "Validation perplexity"),
                      (ax3, "d ln(loss) / d ln(tokens)", "Local power-law slope of train loss"),
                      (ax4, "tokens needed / master's tokens, same loss",
                       "Token stretch vs master (best c with L(t) = L_master(t / c), factor-2 window)")):
    ax.set_xscale("log")
    if ax in (ax1, ax2, ax4): ax.set_yscale("log")
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.set_ylabel(yl, color=INK2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.legend(fontsize=9, frameon=False)
ax1.set_xlim(3e4, 7e8); ax1.set_ylim(2.5, 11)
ax2.set_xlim(5e6, 7e8); ax2.set_ylim(12, 250)
ax3.set_xlim(1e6, 7e8); ax3.set_ylim(-0.5, 0.05); ax3.axhline(0, color=INK2, lw=1)
ax4.set_xlim(1e6, 7e8); ax4.set_ylim(0.5, 4.0); ax4.axhline(1, color=INK2, lw=1)
ax4.set_yticks([0.5, 0.75, 1, 1.5, 2, 3, 4]); ax4.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3", "4"])
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=130, facecolor=SURFACE)
print("wrote", a.out)
