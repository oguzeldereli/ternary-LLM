"""The best runs so far, one per step of progress: validation loss against training tokens (110M model, 300M
tokens, wiki32k). Left: the whole run (log tokens); right: 100M-300M (linear) with the final loss and perplexity.
References dashed; unfinished runs are marked "running".

  python -m scripts.plots.plot_best      -> docs/figures/best_runs.png
"""
import json, math, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
REFS = {"fp32_baseline", "master_tracked"}
RUNS = [  # (run, label, colour)
    ("fp32_baseline", "full precision (fp32 + AdamW), reference", "#757575"),
    ("master_tracked", "master weights (ternary forward, fp32 latent), reference", "#eb6834"),
    ("gvsharp_dry_r1024_s0", "long memory (dry friction) + rank 1024", "#7f0000"),
    ("gvsharp_dryspend_r512_s0", "long memory (dry friction) + spend + rank 512", "#b71c1c"),
    ("gvsharp_b0995_r512_s0", "long memory (decay 0.995) + rank 512", "#0d47a1"),
    ("gvsharp_dryspend_s0", "long memory (dry friction) + spend", "#e53935"),
    ("gvsharp_b0995_s0", "long memory (decay 0.995)", "#1e88e5"),
    ("gvsharp_dry_s0", "long memory (dry friction 1/33)", "#ef6c00"),
    ("gvsharp_rc_s0", "gate + Adam step + sharp (short memory, decay 0.97)", "#c2185b"),
    ("magadd16_wd_qk_lab", "look-ahead + sharp (3 passes per step)", "#9e9e9e"),
    ("gatevnorm_rc_s0", "gate + Adam step + row/col scales", "#2e7d32"),
    ("lm_lowrank256_xb2_100M", "rank-256 momentum + look-ahead x2", "#212121"),
    ("rc_s0", "rank-256 momentum + row/col scales", "#7b1fa2"),
]


def load(d):
    if os.path.exists(f"checkpoints/{d}_lab/metrics.jsonl"): d = f"{d}_lab"
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    va, tps, last = {}, 32768.0, 0
    for l in open(f):
        r = json.loads(l)
        if "loss" in r and "tokens" in r: tps = r["tokens"] / (r["step"] + 1); last = max(last, r["step"])
        if "val_loss" in r: va[r["step"]] = r["val_loss"]
    if not va: return None
    s = np.array(sorted(va))
    fin = None
    for c in (f"checkpoints/{d}/train.log",):
        if os.path.exists(c):
            m = re.findall(r"FINAL val loss ([\d.]+)", open(c).read())
            if m: fin = float(m[-1])
    return (s + 1) * tps, np.array([va[i] for i in s]), last >= 9150, fin


fig, (a, b) = plt.subplots(1, 2, figsize=(22, 9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.15]})
for ax in (a, b):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
data = [(d, lab, c, load(d)) for d, lab, c in RUNS]
data = [x for x in data if x[3] is not None]
data.sort(key=lambda x: x[3][3] or x[3][1][-1])
ENDS = []
for d, lab, c, (t, v, done, fin) in data:
    ref = d in REFS
    ls = "--" if ref else "-"
    end = fin if fin is not None else v[-1]
    tag = f"{end:.3f}, ppl {math.exp(end):.1f}" + ("" if done else f" at {t[-1] / 1e6:.0f}M, running")
    lw = 2.4 if (ref or d.startswith("gvsharp_d") or d.endswith("r512_s0")) else 1.6
    a.plot(t, v, ls, color=c, lw=lw, label=f"{lab}  [{tag}]")
    m = t >= 9.5e7
    b.plot(t[m], v[m], ls, color=c, lw=lw)
    if m.any():   # end labels, nudged apart when two runs end within 0.013 of each other
        y = end
        while any(abs(y - u) < 0.013 for u in ENDS): y += 0.013
        ENDS.append(y)
        b.annotate(f"{end:.3f} ({math.exp(end):.1f})", (t[m][-1], y), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=9.5, color=c)
a.set_xscale("log"); a.set_ylim(2.6, 5.0); a.set_xlim(1e7, 3.2e8)
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
a.set_xlabel("training tokens (log scale)", color=INK2); a.set_ylabel("validation loss (nats per token)", color=INK2)
a.set_title("whole run", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK, title="final loss, perplexity",
         title_fontsize=9)
b.set_xlim(9.5e7, 3.35e8); b.set_ylim(2.65, 3.45)
b.set_xlabel("training tokens (linear)", color=INK2); b.set_ylabel("validation loss (nats per token)", color=INK2)
b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
b.set_title("100M-300M; labels: final loss (perplexity)", color=INK, fontsize=11, loc="left")
p = b.secondary_yaxis("right", functions=(np.exp, np.log))
p.set_ylabel("perplexity", color=INK2); p.tick_params(colors=INK2)
fig.suptitle("Best runs so far: 110M ternary model, 300M tokens of Wikipedia, no full-precision weights "
             "(except the two dashed references)", color=INK, fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/best_runs.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/best_runs.png")
