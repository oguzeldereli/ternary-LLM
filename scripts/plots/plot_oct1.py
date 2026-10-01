"""1 Oct runs: flip selection (flat / inv / cheap / cheap blended in on a cosine), the memory warm-up and undo with
the long-memory recipe (110M), and the 1.3B pair on Myriad.

  python -m scripts.plots.plot_oct1   -> docs/figures/selection_runs.png, memory_undo_runs.png, runs_1b.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TPS = 32768
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M")


def load(*cands):
    """(train steps, trailing mean of 20 logged losses, val steps, val losses) from the first existing metrics file"""
    for c in cands:
        f = f"checkpoints/{c}/metrics.jsonl"
        if os.path.exists(f): break
    else:
        return None
    tr, va = {}, {}
    for l in open(f):
        x = json.loads(l)
        if "val_loss" in x: va[x["step"]] = x["val_loss"]
        elif "loss" in x: tr[x["step"]] = x["loss"]
    s = np.array(sorted(tr)); l = np.array([tr[k] for k in s])
    cs = np.concatenate([[0.0], np.cumsum(l)]); i = np.arange(1, len(l) + 1); lo = np.maximum(i - 20, 0)
    vs = np.array(sorted(va))
    return s, (cs[i] - cs[lo]) / (i - lo), vs, np.array([va[k] for k in vs])


def style(ax, title, xlab="training tokens", ylab="validation loss"):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.xaxis.set_major_formatter(fmt)
    ax.set_title(title, color=INK, fontsize=11, loc="left"); ax.set_xlabel(xlab, color=INK2); ax.set_ylabel(ylab, color=INK2)


def diff(a, b):
    """a's validation minus b's at the steps both have"""
    da, db = dict(zip(a[2], a[3])), dict(zip(b[2], b[3]))
    k = np.array(sorted(set(da) & set(db)))
    return (k + 1) * TPS, np.array([da[x] - db[x] for x in k])


def final(d):
    return f"{d[3][-1]:.4f}" + ("" if d[2][-1] >= 9150 else f" @{(d[2][-1] + 1) * TPS / 1e6:.0f}M, running")


# ---------------- flip selection
M = load("master_tracked")
RULE = {512: load("gvsharp_dryspend_r512_s0_lab"), 256: load("gvsharp_dryspend_s0_lab"),
        128: load("rk128_dryspend_s0_lab"), 64: load("rk64_dryspend_s0_lab")}
SEL = [("flat512_dryspend_s0", "flat (sign only)", "#7cb342", 512),
       ("cheap512_dryspend_s0", "cheap (prefer small v)", "#c2185b", 512),
       ("inv512_dryspend_s0", "inv (prefer small |S|)", "#8e24aa", 512),
       ("cheapcos512_dryspend_s0", "rule blended to cheap on a cosine", "#f9a825", 512)]
CHEAP = [(512, "#c2185b"), (256, "#e57373"), (128, "#f06292"), (64, "#f8bbd0")]
fig, axs = plt.subplots(1, 3, figsize=(24, 7.5), facecolor=SURFACE)
a = axs[0]; style(a, "rank 512: validation loss")
a.plot((M[2] + 1) * TPS, M[3], color="#eb6834", lw=2, label=f"master weights  [{final(M)}]")
a.plot((RULE[512][2] + 1) * TPS, RULE[512][3], color="#1565c0", lw=2.2, label=f"the rule (p ~ |S|)  [{final(RULE[512])}]")
for r, lab, c, _ in SEL:
    d = load(r + "_lab", r)
    a.plot((d[2] + 1) * TPS, d[3], color=c, lw=1.6, label=f"{lab}  [{final(d)}]")
a.set_ylim(2.75, 3.6); a.set_xlim(0, 3.05e8); a.legend(frameon=False, fontsize=9, labelcolor=INK)
b = axs[1]; style(b, "rank 512: minus the rule (same step)", ylab="validation loss minus the rule's")
b.axhline(0, color="#1565c0", lw=1.5)
x, y = diff(M, RULE[512]); b.plot(x, y, color="#eb6834", lw=2, label="master weights")
for r, lab, c, _ in SEL:
    x, y = diff(load(r + "_lab", r), RULE[512]); b.plot(x, y, color=c, lw=1.6, label=lab)
b.set_ylim(-0.2, 0.45); b.set_xlim(0, 3.05e8); b.legend(frameon=False, fontsize=9, labelcolor=INK)
c_ = axs[2]; style(c_, "cheap at every rank, minus the rule at the same rank", ylab="validation loss minus the rule's")
c_.axhline(0, color="#1565c0", lw=1.5)
for rk, col in CHEAP:
    d = load(f"cheap{rk}_dryspend_s0_lab"); x, y = diff(d, RULE[rk])
    c_.plot(x, y, color=col, lw=1.8, label=f"cheap, rank {rk}  [{d[3][-1]:.4f} vs {RULE[rk][3][-1]:.4f}]")
c_.set_ylim(-0.05, 0.3); c_.set_xlim(0, 3.05e8); c_.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("110M, flip selection: same flip count, direction and gate as the rule; only which weights fire changes",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/selection_runs.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/selection_runs.png")

# ---------------- memory warm-up and undo
MEM = [("drywarm512_dryspend_s0", "memory warm-up (friction 0.12 easing to 1/33 over 49M tokens)", "#2e7d32"),
       ("undo512_dryspend_s0", "undo (last step's moves that g and S both call uphill)", "#6d4c41")]
BASE = load("gvsharp_rc_s0_lab")
fig, axs = plt.subplots(1, 3, figsize=(24, 7.5), facecolor=SURFACE)
a = axs[0]; style(a, "training loss, log tokens (trailing mean of 20 logs)", xlab="training tokens (log)", ylab="training loss")
for d, lab, col in ((M, "master weights", "#eb6834"), (RULE[512], "the rule, rank 512 (dry 1/33 from the start)", "#1565c0"),
                    (BASE, "short memory (decay 0.97, no dry)", "#90a4ae")):
    a.plot((d[0] + 1) * TPS, d[1], color=col, lw=1.8, label=lab)
for r, lab, col in MEM:
    d = load(r + "_lab", r); a.plot((d[0] + 1) * TPS, d[1], color=col, lw=1.8, label=lab)
a.set_xscale("log"); a.set_xlim(1e6, 3.1e8); a.set_ylim(2.6, 8); a.legend(frameon=False, fontsize=9, labelcolor=INK)
b = axs[1]; style(b, "validation loss")
for d, lab, col in ((M, "master weights", "#eb6834"), (RULE[512], "the rule, rank 512", "#1565c0")):
    b.plot((d[2] + 1) * TPS, d[3], color=col, lw=2, label=f"{lab}  [{final(d)}]")
for r, lab, col in MEM:
    d = load(r + "_lab", r); b.plot((d[2] + 1) * TPS, d[3], color=col, lw=1.8, label=f"{lab.split(' (')[0]}  [{final(d)}]")
b.set_ylim(2.75, 4.2); b.set_xlim(0, 3.05e8); b.legend(frameon=False, fontsize=9, labelcolor=INK)
c_ = axs[2]; style(c_, "minus the rule (same step)", ylab="validation loss minus the rule's")
c_.axhline(0, color="#1565c0", lw=1.5)
x, y = diff(M, RULE[512]); c_.plot(x, y, color="#eb6834", lw=2, label="master weights")
for r, lab, col in MEM:
    x, y = diff(load(r + "_lab", r), RULE[512]); c_.plot(x, y, "o-", color=col, lw=1.8, ms=3, label=lab.split(" (")[0])
c_.set_xlim(0, 3.05e8); c_.set_ylim(-0.2, 0.15); c_.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("110M, rank 512 recipe: memory that starts short and lengthens; undo with the long memory (both running)",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/memory_undo_runs.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/memory_undo_runs.png")

# ---------------- 1.3B on Myriad, with the 110M / 340M pairs for scale
P = [("110M", load("master_tracked"), RULE[512], "#9ecae1"),
     ("340M", load("big_master_lab", "big_master"), load("big_dryspend_r512_s0_lab"), "#4292c6"),
     ("1.3B", load("x1b_master_myr"), load("x1b_dryspend_r512_s0_myr"), "#08306b")]
fig, axs = plt.subplots(1, 3, figsize=(24, 7.5), facecolor=SURFACE)
a = axs[0]; style(a, "1.3B (d2048_l24): training loss, log tokens (trailing mean of 20 logs)", xlab="training tokens (log)",
                  ylab="training loss")
X = P[2]
a.plot((X[1][0] + 1) * TPS, X[1][1], color="#eb6834", lw=1.8, label="master weights (fp32 latents, AdamW)")
a.plot((X[2][0] + 1) * TPS, X[2][1], color="#08306b", lw=1.8, label="ours, rank 512 (dry + spend)")
a.set_xscale("log"); a.set_xlim(1e6, 3.1e8); a.set_ylim(2.5, 11); a.legend(frameon=False, fontsize=9, labelcolor=INK)
b = axs[1]; style(b, "validation loss at the three sizes (master dashed)")
for nm, m, o, col in P:
    b.plot((m[2] + 1) * TPS, m[3], "--", color=col, lw=1.6, label=f"{nm} master  [{final(m)}]")
    b.plot((o[2] + 1) * TPS, o[3], "-", color=col, lw=2, label=f"{nm} ours r512  [{final(o)}]")
b.set_xlim(0, 3.05e8); b.set_ylim(2.4, 5.5); b.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=1)
c_ = axs[2]; style(c_, "ours minus master (same step), log tokens", xlab="training tokens (log)",
                   ylab="validation loss, ours minus master")
c_.axhline(0, color="#eb6834", lw=1.5)
for nm, m, o, col in P:
    x, y = diff(o, m); c_.plot(x, y, "o-", color=col, lw=1.8, ms=3, label=nm)
c_.set_xscale("log"); c_.set_xlim(5e6, 3.1e8); c_.set_ylim(-0.3, 0.6); c_.legend(frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle("1.3B on Myriad (A100s, running) next to 110M and 340M: the same recipe at rank 512 vs master weights",
             color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/runs_1b.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/runs_1b.png")
