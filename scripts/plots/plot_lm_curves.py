"""LM 10M screens: raw train-loss curves (light) with a smoothed line on top, linear axes,
full view and the last-third zoom; final validation loss in the legend.

  python -m scripts.plots.plot_lm_curves
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# every LM run with 32,768 tokens/step, cut at 10.4M tokens; families share a color, the
# best of each family (lowest mean train loss over its last 20 steps in the window) is bold
FAMILIES = [  # label, color, runs
    ("master weights + AdamW", "#eb6834", ["p2_baseline", "probe_master"]),
    ("momentum (rank 256) + cross-batch look-ahead", "#e53935", ["lm_lowrank256_xb2", "lm_lowrank256_adapt_xb2"]),
    ("cross-batch look-ahead", "#111111", ["la_xb2", "la_xb2_rc", "la_xb1", "la_xb2_r0.125", "overnight_full"]),
    ("same-batch look-ahead", "#c2185b", ["la_fast_fp32tail", "la_fastramp_lr15", "la_fast_fp32tail_r04", "la1", "la2",
                                          "la_r0ramp", "la_rwarm", "armA_cos_la1", "probe_fast"]),
    ("plain flips (rules, schedules, add-ons)", "#1baf7a",
     ["lm_plain", "armA_cosine", "armA_cos_600M", "armA_linear", "armA_cos_r005", "armA_cos_floor", "armA_cos_lr15",
      "armA_cos_lockout", "armA_cos_ef", "armB_absscale", "p3b_acc1", "lr_ctl", "lo_once_t200", "lo_once_t1000",
      "lo_norev_t1000", "ef_probe_a0", "ef_probe_a0.01", "ef_probe_a0.03", "ef_probe_a0.1", "gref1", "gref7",
      "gref10", "gref10_r3x", "gref25", "gref100", "hump", "invmag", "rownorm", "rs_smoke", "rc_plain",
      "rc_plain_lr1e2"]),
    ("frozen random ternary", "#8e8c85", ["lm_frozen"]),
    ("momentum (rank 256), no look-ahead", "#7b3fb8", ["lm_lowrank256", "lm_lowrank256_r04", "lm_lowrank256_r1",
                                                       "lm_lowrank256_adapt"]),
    ("3-bit evidence counter", "#795548", ["ev3_screen"]),
]
MAX_STEP = 316


def load(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r and r["step"] <= MAX_STEP}
    v = [r["val_loss"] for r in rows if r.get("final") and "val_loss" in r and r["step"] <= MAX_STEP + 1]
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    return s, L, (v[-1] if v else None)


def smooth(y, k=15):
    c = np.cumsum(np.insert(y, 0, 0.0))
    w = [min(k, i // 3 + 1) for i in range(len(y))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(y))])


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/archive/lm_curves.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor=SURFACE)
n = 0
for fam, col, runs in FAMILIES:
    D = {d: load(d) for d in runs if os.path.exists(f"checkpoints/{d}/metrics.jsonl")}
    D = {d: v for d, v in D.items() if len(v[0]) > 20}
    best = min(D, key=lambda d: D[d][1][-20:].mean() if D[d][0][-1] >= 290 else 99)
    for d, (st, L, v) in D.items():
        n += 1
        if d == best:
            lab = f"{fam}: best {d}" + (f" (val {v:.3f})" if v else "") + f"  [{len(D)} runs]"
            for ax in (ax1, ax2):
                ax.plot(st, smooth(L), color=col, lw=2.8, label=lab, zorder=5)
        else:
            for ax in (ax1, ax2):
                ax.plot(st, smooth(L), color=col, lw=0.9, alpha=0.45, zorder=2)
print(n, "runs")
ax1.set_ylim(4.6, 10.8); ax1.set_xlim(0, 320)
ax2.set_xlim(200, 318); ax2.set_ylim(4.7, 6.8)
for ax, tt in zip((ax1, ax2), ("Train loss, first 10M tokens of every LM run (15-step mean; bold = best of family)",
                               "Zoom: steps 200-316")):
    ax.set_title(tt, loc="left", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("step (32,768 tokens each)", color=INK2); ax.set_ylabel("loss", color=INK2)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color(GRID)
ax1.legend(fontsize=9.5, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=125, facecolor=SURFACE)
print("wrote", a.out)
