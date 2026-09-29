"""Every swing test on one page (41 steps from one checkpoint, true gradient from 32 batches before each step):
held-out loss change against flips per step, the lag-3 reversal of the true gradient, the share of flips going
uphill, and the weights whose momentum never turns back. Master (AdamW on a float latent) is the reference.

  python -m scripts.plots.plot_swing      -> docs/figures/swing_tests.png
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "checkpoints/momentum_mech"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ARMS = [  # (label, file, colour)
    ("master (AdamW, float latent)", "master_wave_5000.txt", "#eb6834"),
    ("old rule", "wave4500_norm_own_rev4090.txt", "#111111"),
    ("old rule, 0.62x rate", "wave4500_rate062_4090.txt", "#757575"),
    ("old rule, memory 0.8", "wave4500_b08.txt", "#9e9e9e"),
    ("fixed divisor", "wave4500_norm_fixed_4090.txt", "#bdbdbd"),
    ("sign gate", "wave4500_full_gate.txt", "#1565c0"),
    ("Adam step (vnorm)", "wave4500_full_vnorm0.99.txt", "#00897b"),
    ("gate + Adam step", "damp_gatevnorm_nola4500.txt", "#2e7d32"),
    ("undo (old rule + undo)", "wave4500_fu4_plain_undo_lab.txt", "#7b1fa2"),
    ("tanh flip chance, beta 0.97", "wave4500_tanh_b097_4090.txt", "#f48fb1"),
    ("yours: beta 1, cap 1x start, tanh", "wave4500_user_ncap1_4090.txt", "#c2185b"),
    ("yours: beta 1, cap 3 x grad", "wave4500_user_gcap3_laptop.txt", "#ad1457"),
    ("yours: beta 1, cap 1 x grad", "wave4500_user_gcap1_laptop.txt", "#880e4f"),
    ("yours: beta 0.97 + cap + tanh", "wave4500_fu1_b097_cap_tanh_4090.txt", "#fbc02d"),
    ("yours + undo", "wave4500_fu2_b097_cap_tanh_undo_4090.txt", "#f57f17"),
    ("yours, beta 0.9 + undo", "wave4500_fu3_b09_cap_tanh_undo_4090.txt", "#ff8f00"),
    ("dry friction, vector", "wave4500_dry_vec_lab.txt", "#6d4c41"),
    ("dry friction, per weight", "wave4500_dry_w_lab.txt", "#8d6e63"),
    ("asymmetric gravity 0.5", "wave4500_grav05_4090.txt", "#00acc1"),
    ("asymmetric gravity 0.8", "wave4500_grav08_4090.txt", "#4dd0e1"),
]


def parse(f):
    p = os.path.join(D, f)
    if not os.path.exists(p): p = os.path.join(D, "lab", f)
    if not os.path.exists(p): return None
    s = open(p).read(); r = {}
    m = re.search(r"held-out loss [\d.]+ -> [\d.]+ \(([+-][\d.]+)\).*?(?:flips|trit changes) per step:.*?mean (\d+)k", s)
    if not m: return None
    r["dL"], r["flips"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^\s+3\s+([+-][\d.]+)\s+([+-][\d.]+)", s, re.M); r["lag3"] = float(m.group(1)) if m else np.nan
    m = re.search(r"^\s+1\s+([+-][\d.]+)\s+([+-][\d.]+)", s, re.M); r["mnext"] = float(m.group(2)) if m else np.nan
    m = re.search(r"(?:flips|trit changes) that move uphill on the true gradient: mean ([\d.]+)%", s)
    r["up"] = float(m.group(1)) if m else np.nan
    m = re.search(r"never(?: within the window)?:? ([\d.]+)%", s); r["never"] = float(m.group(1)) if m else np.nan
    return r


rows = [(lab, c, parse(f)) for lab, f, c in ARMS]
rows = [(lab, c, r) for lab, c, r in rows if r]
fig, axs = plt.subplots(2, 2, figsize=(20, 13), facecolor=SURFACE)
for ax in axs.flat:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
a = axs[0, 0]
for lab, c, r in rows:
    a.scatter(r["flips"], r["dL"], s=90 if lab.startswith("master") else 55, color=c, zorder=3,
              edgecolor=INK if lab.startswith("master") else "none")
    a.annotate(lab, (r["flips"], r["dL"]), xytext=(5, 3), textcoords="offset points", fontsize=8.5, color=c)
a.axhline(0, color=INK2, lw=0.8)
a.set_xlabel("flips (trit changes) per step, thousands", color=INK2); a.set_ylabel("held-out loss change over 41 steps", color=INK2)
a.set_title("loss gained against how much it moved (fewer flips favour the 41-step number)", color=INK, fontsize=11, loc="left")


def bars(ax, key, title, ref=None, fmt="{:+.2f}"):
    rr = [(lab, c, r[key]) for lab, c, r in rows if not np.isnan(r[key])]
    y = np.arange(len(rr))
    ax.barh(y, [v for *_, v in rr], color=[c for _, c, _ in rr])
    ax.set_yticks(y); ax.set_yticklabels([lab for lab, *_ in rr], fontsize=9); ax.invert_yaxis()
    for i, (_, _, v) in enumerate(rr): ax.annotate(fmt.format(v), (v, i), xytext=(3, 0), textcoords="offset points",
                                                   va="center", fontsize=8.5, color=INK2)
    if ref is not None: ax.axvline(ref, color="#eb6834", lw=1.4, ls="--")
    ax.set_title(title, color=INK, fontsize=11, loc="left")


mref = next((r for lab, _, r in rows if lab.startswith("master")), None)
bars(axs[0, 1], "lag3", "the swing: cos(true gradient now, 3 steps later); negative = it reverses", mref and mref["lag3"])
bars(axs[1, 0], "up", "flips going uphill on the true gradient when made (%)", mref and mref["up"], "{:.1f}")
bars(axs[1, 1], "never", "weights whose momentum never turns back after the gradient reverses (%)", mref and mref["never"], "{:.1f}")
fig.suptitle("Swing tests: 41 steps from one checkpoint (nola_lab @4500; master: curve_master @5000). Dashed = master",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/swing_tests.png", dpi=105, facecolor=SURFACE)
print("wrote docs/figures/swing_tests.png", len(rows), "arms")
