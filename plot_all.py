"""Every run so far on two log-log panels: long runs, and the 300-step alpha screens.

  python3 plot_all.py --out docs/all_runs.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MASTER, BEST, LA = "#eb6834", "#1baf7a", "#c2185b"

# (dir, label, style) -- style: master | best | la | other
LONG = [
    ("p2_baseline",      "master weights + AdamW (ceiling)",    "master"),
    ("armA_cos_600M",    "flips, cosine rate, 600M",             "best"),
    ("armA_cosine",      "flips, cosine rate (reference)",       "best"),
    ("armA_linear",      "flips, linear rate",                   "other"),
    ("armA_cos_r005",    "flips, cosine from 4x lower rate",     "other"),
    ("armA_cos_lockout", "flips, cosine + lockout",              "other"),
    ("armA_cos_floor",   "flips, cosine to 0.02% floor",         "other"),
    ("p3b_acc1",         "flips, constant rate",                 "other"),
    ("p3b_acc4",         "flips, constant rate, 4x batch",       "other"),
    ("armA_cos_ef",      "flips, cosine + error feedback",       "other"),
    ("armB_absscale",    "flips, frozen threshold scale",        "other"),
    ("lr_ctl",           "control, 1000 steps",                  "other"),
    ("lo_once_t200",     "flip-once lockout (t=200)",            "other"),
    ("lo_once_t1000",    "flip-once lockout (t=1000)",           "other"),
    ("lo_norev_t1000",   "no-reversal lockout (t=1000)",         "other"),
    ("armA_cos_lr15",    "cosine, 1.5x lr",                      "other"),
    ("ef_probe_a0",      "EF probe a=0",                         "other"),
    ("ef_probe_a0.01",   "EF probe a=0.01",                      "other"),
    ("ef_probe_a0.03",   "EF probe a=0.03",                      "other"),
    ("ef_probe_a0.1",    "EF probe a=0.1",                       "other"),
]
SCREEN = [
    ("p2_baseline",   "master weights (ceiling)",      "master"),
    ("armA_cosine",   "ramp g_ref=3 (reference)",      "best"),
    ("la1",           "LOOK-AHEAD, 1 pass",            "la"),
    ("la2",           "LOOK-AHEAD, 2 passes",          "la"),
    ("gref1",         "g_ref=1",                       "other"),
    ("gref7",         "g_ref=7",                       "other"),
    ("gref10",        "g_ref=10",                      "other"),
    ("gref10_r3x",    "g_ref=10, 3x rate",             "other"),
    ("gref25",        "g_ref=25",                      "other"),
    ("gref100",       "g_ref=100",                     "other"),
    ("rownorm",       "per-row normalisation",         "other"),
    ("ev3_screen",    "3-bit evidence (not stateless)", "other"),
    ("invmag",        "reverse magnitude",             "other"),
    ("hump",          "hump in |g|",                   "other"),
    ("rs_smoke",      "greedy rate search",            "other"),
]


def load(d):
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f)]
    tr = [r for r in rows if "loss" in r and "tokens" in r]
    if len(tr) < 5: return None
    t = np.array([r["tokens"] for r in tr], float); L = np.array([r["loss"] for r in tr])
    v = [r for r in rows if "val_loss" in r]
    return t, L, (np.exp(v[-1]["val_loss"]) if v else None), np.array([r["step"] for r in tr])


def smooth(L, k):
    c = np.cumsum(np.insert(L, 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i - k + 1)]) / (i + 1 - max(0, i - k + 1)) for i in range(len(L))])


def alpha(t, Ls, s):
    m = (s >= 100) & (s <= 300)
    return np.polyfit(np.log(t[m]), np.log(Ls[m]), 1)[0] if m.sum() > 50 and s.max() >= 290 else None


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/all_runs.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), facecolor=SURFACE)
others = plt.get_cmap("tab20")

for ax, runs, title in ((ax1, LONG, "All training runs (train loss, 100-step smoothing)"),
                        (ax2, SCREEN, "300-step α screens (train loss, 25-step smoothing, α over steps 100-300)")):
    oi = 0
    for d, lab, st in runs:
        r = load(d)
        if r is None: continue
        t, L, ppl, s = r
        if ax is ax2:
            keep = s <= 300; t, L, s = t[keep], L[keep], s[keep]
        Ls = smooth(L, 100 if ax is ax1 else 25)
        if st == "master": c, lw, z = MASTER, 2.6, 5
        elif st == "best": c, lw, z = BEST, 2.6, 5
        elif st == "la": c, lw, z = LA, 3.0, 6
        else: c, lw, z = others(oi % 20), 1.2, 3; oi += 1
        ls = "--" if (st == "la" and d == "la2") else "-"
        if ax is ax1:
            extra = f"  ppl {ppl:.1f}" if ppl else ""
            extra += f"  [{t[-1]/1e6:.0f}M tok]"
        else:
            al = alpha(t, Ls, s)
            extra = f"  α {al:+.3f}" if al is not None else (f"  (running, step {s.max()})" if d == "la2" else f"  (stopped at step {s.max()})")
        ax.plot(t, Ls, color=c, lw=lw, ls=ls, zorder=z, label=lab + extra)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_xlabel("tokens", color=INK2); ax.set_ylabel("train loss", color=INK2)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
ax1.set_ylim(2.3, 11); ax2.set_xlim(2e5, 1.05e7); ax2.set_ylim(5.0, 11)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=130, facecolor=SURFACE)
print("wrote", a.out)
