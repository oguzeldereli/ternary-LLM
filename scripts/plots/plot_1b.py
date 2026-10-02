"""1.3B (d2048_l24) on Myriad: master and our old rule (dry friction + spend, rank 512) at peak lr 7.5e-4, with the
stopped lr 1.5e-3 pair faded; and the gap to master at each size (110M / 340M / 1.3B) for the old rule and for --ts.

  python -m scripts.plots.plot_1b      -> docs/figures/runs_1b.png
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TPS = 32768
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M")


def load(r):
    """(train steps, trailing mean of 20 logged losses, {step: val}) from the first existing file"""
    for f in (f"checkpoints/{r}_myr/metrics.jsonl", f"checkpoints/{r}_lab/metrics.jsonl", f"checkpoints/{r}/metrics.jsonl"):
        if os.path.exists(f): break
    else:
        f = f"checkpoints/{r}_lab/night.txt"
        if not os.path.exists(f): return None
        va = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'"step": (\d+), "val_loss": ([\d.]+)', open(f).read())}
        return np.array([]), np.array([]), va
    tr, va = {}, {}
    for l in open(f):
        x = json.loads(l)
        if "val_loss" in x: va[x["step"]] = x["val_loss"]
        elif "loss" in x: tr[x["step"]] = x["loss"]
    s = np.array(sorted(tr)); l = np.array([tr[k] for k in s])
    cs = np.concatenate([[0.0], np.cumsum(l)]); i = np.arange(1, len(l) + 1); lo = np.maximum(i - 20, 0)
    return s, (cs[i] - cs[lo]) / (i - lo), va


def style(ax, title, ylab, logx=True):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.xaxis.set_major_formatter(fmt)
    if logx: ax.set_xscale("log")
    ax.set_title(title, color=INK, fontsize=11, loc="left"); ax.set_ylabel(ylab, color=INK2)
    ax.set_xlabel("training tokens" + (" (log)" if logx else ""), color=INK2)


def vline(ax, d, *a, **k):
    s = np.array(sorted(d)); ax.plot((s + 1) * TPS, [d[i] for i in s], *a, **k)


def tag(d):
    s = max(d); return f"{d[s]:.4f}" + ("" if s >= 9150 else f" @{(s + 1) * TPS / 1e6:.0f}M, running")


def gap(o, m):
    k = np.array(sorted(s for s in o if s in m and s >= 250)); return (k + 1) * TPS, np.array([o[s] - m[s] for s in k])


M1, O1 = load("x1b_master_lr75"), load("x1b_dryspend_r512_lr75")
M1h, O1h = load("x1b_master"), load("x1b_dryspend_r512_s0")
fig, axs = plt.subplots(1, 3, figsize=(25, 8), facecolor=SURFACE)
a = axs[0]; style(a, "1.3B: training loss (trailing mean of 20 logs)", "training loss")
a.plot((M1[0] + 1) * TPS, M1[1], color="#eb6834", lw=1.8, label="master weights, lr 7.5e-4")
a.plot((O1[0] + 1) * TPS, O1[1], color="#08306b", lw=1.8, label="our old rule (dry + spend, rank 512), lr 7.5e-4")
a.plot((M1h[0] + 1) * TPS, M1h[1], color="#eb6834", lw=1, alpha=0.35, label="master, lr 1.5e-3 (diverged, stopped)")
a.plot((O1h[0] + 1) * TPS, O1h[1], color="#08306b", lw=1, alpha=0.35, label="old rule, lr 1.5e-3 (stopped)")
a.set_xlim(1e6, 3.1e8); a.set_ylim(2.4, 9); a.legend(frameon=False, fontsize=9, labelcolor=INK)
b = axs[1]; style(b, "validation loss: 1.3B pair, with the 340M and 110M masters", "validation loss", logx=False)
vline(b, M1[2], "-", color="#eb6834", lw=2.2, label=f"1.3B master  [{tag(M1[2])}]")
vline(b, O1[2], "-", color="#08306b", lw=2.2, label=f"1.3B our old rule  [{tag(O1[2])}]")
for r, lab, c in (("big_master", "340M master", "#f4a582"), ("master_tracked", "110M master", "#fddbc7")):
    d = load(r); vline(b, d[2], "--", color=c, lw=1.6, label=f"{lab}  [{tag(d[2])}]")
b.set_xlim(0, 3.05e8); b.set_ylim(2.5, 4.0); b.legend(frameon=False, fontsize=9, labelcolor=INK)
c = axs[2]; style(c, "gap to master at each size (same step)", "validation loss minus master's")
c.axhline(0, color="#eb6834", lw=1.5)
PAIRS = [("110M old rule r512", "master_tracked", "gvsharp_dryspend_r512_s0", "#9ecae1", "-"),
         ("340M old rule r512", "big_master", "big_dryspend_r512_s0", "#4292c6", "-"),
         ("1.3B old rule r512 (lr 7.5e-4)", "x1b_master_lr75", "x1b_dryspend_r512_lr75", "#08306b", "-"),
         ("110M --ts r512 (tau 1000)", "master_tracked", "ts16rs512tau1000_s0", "#80cbc4", "--"),
         ("110M --ts r512, steps x ratio^0.5", "master_tracked", "ts16rs512tau1000an05_s0", "#00897b", "--"),
         ("340M --ts r512 (tau 1000)", "big_master", "big_ts16rs512tau1000_s0", "#00695c", "--")]
for lab, m, o, col, ls in PAIRS:
    dm, do = load(m), load(o)
    if not dm or not do: continue
    x, y = gap(do[2], dm[2]); c.plot(x, y, ls, color=col, lw=1.9, marker="o", ms=2.5, label=f"{lab}  [{y[-1]:+.3f}]")
c.set_xlim(5e6, 3.1e8); c.set_ylim(-0.25, 1.0); c.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("1.3B on Myriad (1 A100 each): master 2.5868; our old rule at the same lr still +0.2 behind at 130M tokens; "
             "--ts at 110M / 340M for comparison", color=INK, fontsize=13, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/runs_1b.png", dpi=90, facecolor=SURFACE)
print("wrote docs/figures/runs_1b.png")
