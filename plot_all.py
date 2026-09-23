"""Every run so far on one log-log panel, grouped: a shaded band spans each group's
runs (min-max of smoothed train loss at each token count, where 2+ runs exist) and a
solid line shows the group's best run (chosen by hand in GROUPS: the one with the best
final validation result among its longest runs).

  python3 plot_all.py --out docs/all_runs.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

# (group label, color, best run shown as line, [member runs])
GROUPS = [
    ("master weights + AdamW (ceiling)", "#eb6834", "p2_baseline", ["p2_baseline"]),
    ("stateless look-ahead filter", "#c2185b", "armA_cos_la1",
     ["armA_cos_la1", "la1", "la2", "la_rwarm"]),
    ("stateless flips: rate schedules, batch, lr", "#1baf7a", "armA_cos_600M",
     ["armA_cosine", "armA_cos_600M", "armA_linear", "armA_cos_r005", "armA_cos_floor",
      "p3b_acc1", "p3b_acc4", "armA_cos_lr15", "lr_ctl"]),
    ("stateless flips: flip-rule shape (g_ref, row-norm, hump, reverse, rate search)",
     "#2a78d6", "gref7",
     ["gref1", "gref7", "gref10", "gref10_r3x", "gref25", "gref100", "rownorm", "invmag",
      "hump", "rs_smoke"]),
    ("stateless flips + add-ons (lockout, error feedback, frozen scale)", "#8e8c85",
     "armA_cos_lockout",
     ["armA_cos_lockout", "lo_once_t200", "lo_once_t1000", "lo_norev_t1000", "armA_cos_ef",
      "ef_probe_a0", "ef_probe_a0.01", "ef_probe_a0.03", "ef_probe_a0.1", "armB_absscale"]),
    ("3-bit evidence counter (not stateless)", "#7b3fb8", "ev3_screen", ["ev3_screen"]),
]
GRIDX = np.logspace(np.log10(3.3e4), np.log10(6.1e8), 400)


def load(d):
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f)]
    tr = {}
    for r in rows:                       # last record per step wins (resumes re-log)
        if "loss" in r and "tokens" in r: tr[r["step"]] = r
    if len(tr) < 10: return None
    s = sorted(tr)
    t = np.array([tr[i]["tokens"] for i in s], float)
    L = np.array([tr[i]["loss"] for i in s])
    k = 50
    c = np.cumsum(np.insert(L, 0, 0.0))
    Ls = np.array([(c[i + 1] - c[max(0, i - k + 1)]) / (i + 1 - max(0, i - k + 1)) for i in range(len(L))])
    v = [r for r in rows if "val_loss" in r]
    return t, Ls, (np.exp(v[-1]["val_loss"]) if v else None)


def on_grid(t, Ls):
    y = np.interp(np.log(GRIDX), np.log(t), np.log(Ls), left=np.nan, right=np.nan)
    return np.exp(y)


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/all_runs.png")
a = ap.parse_args()
fig, ax = plt.subplots(figsize=(15, 9), facecolor=SURFACE)
for label, color, best, members in GROUPS:
    curves = {d: load(d) for d in members}
    curves = {d: r for d, r in curves.items() if r is not None}
    if not curves: continue
    G = np.array([on_grid(t, Ls) for t, Ls, _ in curves.values()])
    n = np.sum(~np.isnan(G), axis=0)
    lo, hi = np.nanmin(np.where(n > 0, G, np.inf), axis=0), np.nanmax(np.where(n > 0, G, -np.inf), axis=0)
    m = n >= 2
    if m.any():
        ax.fill_between(GRIDX, np.where(m, lo, np.nan), np.where(m, hi, np.nan),
                        color=color, alpha=0.18, lw=0)
    t, Ls, ppl = curves[best]
    tag = f"  [best: {best}" + (f", val ppl {ppl:.1f}" if ppl else "") + f", {t[-1]/1e6:.0f}M tok]"
    ax.plot(t, Ls, color=color, lw=2.6 if len(members) > 1 else 2.2,
            label=f"{label} — {len(curves)} run{'s' if len(curves) > 1 else ''}{tag}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(3e4, 7e8); ax.set_ylim(2.5, 11)
ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
ax.set_xlabel("tokens", color=INK2); ax.set_ylabel("train loss (50-step avg)", color=INK2)
ax.set_title("All runs, grouped: band = range of the group's runs, line = its best run",
             color=INK, fontsize=12, loc="left")
for sp in ax.spines.values(): sp.set_color(GRID)
ax.tick_params(colors=INK2)
ax.legend(fontsize=9, frameon=False, loc="lower left")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=130, facecolor=SURFACE)
print("wrote", a.out)
