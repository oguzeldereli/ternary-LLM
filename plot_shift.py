"""Flip runs with their first fast drop cut off, overlaid on the start of master.

Every flip run falls quickly for the first ~10 steps (to ~9.2) and then plateaus near
10^5.5 tokens before the main descent. Everything before the cut T0 is dropped and the
rest is re-based onto master two ways:
  left:  tokens counted from the cut (t - T0 + one step), so all curves start together
  right: shifted along x so each run's loss at the cut sits on master's curve at the
         same loss (compares "how fast from here" at equal loss)
Master and the evidence run (no early drop) are drawn unshifted.

  python3 plot_shift.py --out docs/all_runs_shifted.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
T0 = 10 ** 5.5
STEP_TOK = 32768
GROUPS = [
    ("master weights + AdamW (ceiling)", "#eb6834", "p2_baseline", ["p2_baseline"], False),
    ("stateless look-ahead filter", "#c2185b", "armA_cos_la1",
     ["armA_cos_la1", "la1", "la2", "la_rwarm"], True),
    ("stateless flips: rate schedules, batch, lr", "#1baf7a", "armA_cos_600M",
     ["armA_cosine", "armA_cos_600M", "armA_linear", "armA_cos_r005", "armA_cos_floor",
      "p3b_acc1", "p3b_acc4", "armA_cos_lr15", "lr_ctl"], True),
    ("stateless flips: flip-rule shape", "#2a78d6", "gref7",
     ["gref1", "gref7", "gref10", "gref10_r3x", "gref25", "gref100", "rownorm", "invmag",
      "hump", "rs_smoke"], True),
    ("stateless flips + add-ons", "#8e8c85", "armA_cos_lockout",
     ["armA_cos_lockout", "lo_once_t200", "lo_once_t1000", "lo_norev_t1000", "armA_cos_ef",
      "ef_probe_a0", "ef_probe_a0.01", "ef_probe_a0.03", "ef_probe_a0.1", "armB_absscale"], True),
    ("3-bit evidence counter (not stateless, unshifted)", "#7b3fb8", "ev3_screen",
     ["ev3_screen"], False),
]
GRIDX = np.logspace(np.log10(3.0e4), np.log10(6.5e8), 400)


def load(d):
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    tr = {}
    for l in open(f):
        r = json.loads(l)
        if "loss" in r and "tokens" in r: tr[r["step"]] = r
    if len(tr) < 10: return None
    s = sorted(tr)
    t = np.array([tr[i]["tokens"] for i in s], float)
    L = np.array([tr[i]["loss"] for i in s])
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(50, i // 5 + 1) for i in range(len(L))]
    return t, np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


master = load("p2_baseline")


def shift(t, Ls, mode):
    k = t >= T0
    t, Ls = t[k], Ls[k]
    if mode == "tokens":
        return t - t[0] + STEP_TOK, Ls
    # loss-matched: first master token where master's loss <= this run's loss at the cut
    mt, mL = master
    j = np.argmax(mL <= Ls[0])
    return t - t[0] + mt[j], Ls


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/all_runs_shifted.png")
a = ap.parse_args()
fig, axes = plt.subplots(1, 2, figsize=(22, 9), facecolor=SURFACE)
for ax, mode, title in ((axes[0], "tokens", f"Tokens counted from the cut at 10^5.5 (flip runs); master from 0"),
                        (axes[1], "loss", "Flip runs slid along x so their loss at the cut sits on master's curve")):
    for label, color, best, members, shifted in GROUPS:
        curves = {d: load(d) for d in members}
        curves = {d: r for d, r in curves.items() if r is not None}
        if not curves: continue
        if shifted:
            curves = {d: shift(t, Ls, mode) for d, (t, Ls) in curves.items()}
        G = np.array([np.exp(np.interp(np.log(GRIDX), np.log(t), np.log(Ls), left=np.nan, right=np.nan))
                      for t, Ls in curves.values()])
        n = np.sum(~np.isnan(G), axis=0); m = n >= 2
        if m.any():
            lo = np.nanmin(np.where(n > 0, G, np.inf), axis=0)
            hi = np.nanmax(np.where(n > 0, G, -np.inf), axis=0)
            ax.fill_between(GRIDX, np.where(m, lo, np.nan), np.where(m, hi, np.nan),
                            color=color, alpha=0.18, lw=0)
        t, Ls = curves[best]
        ax.plot(t, Ls, color=color, lw=2.6, label=f"{label} ({len(curves)} run{'s' if len(curves) > 1 else ''}, best: {best})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3e4, 7e8); ax.set_ylim(2.5, 11)
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens (shifted for flip runs)", color=INK2)
    ax.set_ylabel("train loss (adaptive trailing avg)", color=INK2)
    ax.set_title(title, color=INK, fontsize=11.5, loc="left")
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.legend(fontsize=8.5, frameon=False, loc="lower left")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=120, facecolor=SURFACE)
print("wrote", a.out)
