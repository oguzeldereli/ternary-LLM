"""Night 1-2 Oct: what master cannot do without (master from scratch with one ingredient removed), and our rule rebuilt
on master's pattern (--ts: short momentum, long accumulator, fire at a threshold), with its rank sweep and 340M run.
Validation loss minus master's at the same step (110M), log tokens.

  python -m scripts.plots.plot_night_oct2 [MYRIAD_STATUS_DUMP]   -> docs/figures/night_oct2.png
  (lab / 4090 runs: checkpoints/NAME_lab/night.txt; Myriad runs: blocks "== NAME" of the dump, if given)
"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TPS = 32768
VAL = re.compile(r'"step": (\d+), "val_loss": ([\d.]+)')


def metrics(path):
    d = {}
    for l in open(path):
        if '"val_loss"' in l:
            x = json.loads(l); d[x["step"]] = x["val_loss"]
    return d


def lines(ls):
    return {int(m.group(1)): float(m.group(2)) for m in map(VAL.search, ls) if m}


MYR = {}
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    for blk in open(sys.argv[1]).read().split("== ")[1:]:
        name, *ls = blk.splitlines()
        MYR[name] = lines(ls)


def run(name):
    if name in MYR: return MYR[name]
    f = f"checkpoints/{name}_lab/night.txt"
    return lines(open(f).read().splitlines()) if os.path.exists(f) else {}


M = metrics("checkpoints/master_tracked/metrics.jsonl")
R512 = metrics("checkpoints/gvsharp_dryspend_r512_s0_lab/metrics.jsonl")
RFULL = metrics("checkpoints/gvsharp_dryspend_r1024_s0_lab/metrics.jsonl")
BM = metrics("checkpoints/big_master_lab/metrics.jsonl")
B512 = metrics("checkpoints/big_dryspend_r512_s0_lab/metrics.jsonl")


def diff(d, ref):
    k = np.array(sorted(s for s in d if s in ref and s >= 250))
    return (k + 1) * TPS, np.array([d[s] - ref[s] for s in k])


def label(name, d, lab):
    if not d: return lab
    s = max(d)
    return f"{lab}  [{d[s]:.4f}{'' if s >= 9150 else f' @{(s + 1) * TPS / 1e6:.0f}M'}]"


def style(ax, title, ylab="validation loss minus master's (same step)"):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2); ax.set_xscale("log"); ax.set_xlim(7e6, 3.1e8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.axhline(0, color="#eb6834", lw=1.6)
    ax.set_title(title, color=INK, fontsize=11, loc="left"); ax.set_xlabel("training tokens (log)", color=INK2)
    ax.set_ylabel(ylab, color=INK2)


def draw(ax, items, ref=M):
    for name, lab, col, ls in items:
        d = name if isinstance(name, dict) else run(name)
        if not d: continue
        x, y = diff(d, ref)
        ax.plot(x, y, ls, color=col, lw=1.8, label=label(name if isinstance(name, str) else "", d, lab))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")


fig, axs = plt.subplots(2, 2, figsize=(22, 14), facecolor=SURFACE)
a = axs[0, 0]; style(a, "master from scratch with one ingredient removed (orange line: master 2.7513)")
draw(a, [(R512, "our old rule, rank 512", "#1565c0", "-"),
         ("mx_snap", "no leftover offset after a crossing (snap)", "#c2185b", "-"),
         ("mx_leak300", "offset forgets, tau 300 (our momentum's memory)", "#8e24aa", "-"),
         ("mx_leak1000", "offset forgets, tau 1000", "#ba68c8", "--"),
         ("mx_factv", "factored second moment (ours)", "#7cb342", "--"),
         ("mx_rank512", "first moment rank 512", "#aed581", "--"),
         ("mx_gate", "+ our sign gate", "#00897b", "--"),
         ("mx_b2999", "beta2 0.999 (control)", "#90a4ae", ":"),
         ("mx_b1997b2", "long first moment (beta1 0.997, beta2 0.999)", "#6d4c41", "-"),
         ("mx_lag02", "rate-limited firing (prob. 0.02 / step, like our flips)", "#f9a825", "-")])
a.set_ylim(-0.1, 0.5)
b = axs[0, 1]; style(b, "--ts: short momentum + long accumulator + fire at a threshold (110M)")
draw(b, [(R512, "our old rule, rank 512", "#1565c0", "-"), (RFULL, "our old rule, full rank", "#0d1b4c", "-"),
         ("ts16_s0", "ranks 512/128, tau 300", "#bdbdbd", "-"),
         ("ts16tau1000_s0", "ranks 512/128, tau 1000", "#9e9e9e", "-"),
         ("ts16rs512_s0", "ranks 512/512, tau 300", "#80cbc4", "-"),
         ("ts16rs512tau1000_s0", "ranks 512/512, tau 1000", "#26a69a", "-"),
         ("ts16rs512tau1000an05_s0", "ranks 512/512, tau 1000, steps x ratio^0.5", "#00695c", "-"),
         ("ts16rs512la1000_s0", "ranks 512/512, tau 1000, leak x ratio", "#2e7d32", "--"),
         ("ts16full_tau1000_s0", "full rank, tau 1000", "#e65100", "-"),
         ("ts16fullla1000_s0", "full rank, tau 1000, leak x ratio", "#bf360c", "--"),
         ("tsnoflip_s0", "control: no flips (float extras only)", "#424242", ":")])
b.set_ylim(-0.2, 0.45)
c = axs[1, 0]; style(c, "--ts rank sweep (tau 1000, steps x ratio): accumulator / short momentum")
draw(c, [(R512, "our old rule, rank 512", "#1565c0", "-"),
         ("ts16full_tau1000_s0", "full / full", "#e65100", "-"),
         ("ts16rs1024tau1000_s0", "512 / full", "#fb8c00", "-"),
         ("ts16rs512tau1000_s0", "512 / 512", "#26a69a", "-"),
         ("ts16r512rs256tau1000_s0", "512 / 256", "#5c6bc0", "-"),
         ("ts16r256rs512tau1000_s0", "256 / 512", "#7986cb", "--"),
         ("ts16r256rs256tau1000_s0", "256 / 256", "#9575cd", "-"),
         ("ts16r128rs512tau1000_s0", "128 / 512", "#ce93d8", "--"),
         ("ts16r128rs128tau1000_s0", "128 / 128", "#f48fb1", "-"),
         ("ts16r64rs64tau1000_s0", "64 / 64", "#f8bbd0", "-")])
c.set_ylim(-0.2, 0.3)
d_ = axs[1, 1]; style(d_, "340M: --ts ranks 512/512, tau 1000, minus 340M master (2.6626)",
                      ylab="validation loss minus 340M master's")
draw(d_, [(B512, "our old rule at 340M, rank 512", "#1565c0", "-"),
          ("big_ts16rs512tau1000_s0", "--ts ranks 512/512, tau 1000 (4090)", "#26a69a", "-")], ref=BM)
d_.set_ylim(-0.3, 0.8)
fig.suptitle("Night 1-2 Oct: master needs a short momentum, a ~1000-step integrator and immediate firing; "
             "our rule rebuilt that way passes master", color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/night_oct2.png", dpi=90, facecolor=SURFACE)
print("wrote docs/figures/night_oct2.png")
