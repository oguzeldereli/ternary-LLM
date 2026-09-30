"""Where the remaining gap to master weights is, at 300M tokens (110M model): held-out loss minus master's, by how
often the (previous, target) pair occurs in the training set (left) and by position in the context (middle), and the
ordered-copy (induction) gain (right). Read from the loss_by_freq / loss_by_pos outputs in checkpoints/momentum_mech/.

  python -m scripts.plots.plot_gaps      -> docs/figures/gaps.png
"""
import glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
D = "checkpoints/momentum_mech"
RUNS = [  # (run, label, colour); the step is the final one (9154)
    ("gvsharp_rc_s0", "short memory (gate + Adam step + sharp)", "#c2185b"),
    ("gvsharp_b099_s0", "decay 0.99", "#90caf9"),
    ("gvsharp_b0995_s0", "decay 0.995", "#1e88e5"),
    ("gvsharp_dry_s0", "dry friction", "#ef6c00"),
    ("gvsharp_dryspend_s0", "dry friction + spend", "#e53935"),
    ("gvsharp_dryspend_r512_s0", "dry friction + spend + rank 512", "#7f0000"),
]
FREQ, POS = {}, {}
for f in glob.glob(f"{D}/lbf_*laptop.txt"):
    for l in open(f):
        m = re.match(r"(\S+) @(\d+)\s*\|\s+([\d. ]+)\|\s+([\d. ]+)\|\s*([\d.]+)", l)
        if m and m.group(2) == "9154": FREQ[m.group(1)] = [float(x) for x in m.group(4).split()]
for f in glob.glob(f"{D}/lbp_*laptop.txt"):
    for l in open(f):
        m = re.match(r"(\S+) @(\d+) \(\d+M\)\s+([\d. ]+?)\s+([+-][\d.]+)\s*$", l)
        if m and m.group(2) == "9154":
            v = [float(x) for x in m.group(3).split()]
            POS[m.group(1)] = (v[:5], float(m.group(4)))
M = "master_tracked"
share = [1.9, 4.9, 11.0, 22.7, 27.7, 31.8]            # % of held-out positions per pair bucket (loss_by_freq header)
PB = ["0", "1-9", "10-99", "100-999", "1k-10k", ">10k"]
QB = ["0-1", "2-15", "16-127", "128-511", "512+"]

fig, axs = plt.subplots(1, 3, figsize=(22, 7.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1.1, 0.8]})
for ax in axs:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8, axis="y")
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=INK2)
n = len(RUNS); w = 0.8 / n
a = axs[0]
for i, (r, lab, c) in enumerate(RUNS):
    if r not in FREQ: continue
    gap = np.array(FREQ[r]) - np.array(FREQ[M])
    a.bar(np.arange(6) + (i - n / 2 + 0.5) * w, gap, w, color=c, label=lab)
a.set_xticks(np.arange(6)); a.set_xticklabels([f"{b}\n{s:.0f}%" for b, s in zip(PB, share)])
a.set_xlabel("times the (previous, target) pair occurs in the 400M-token training set\n(% = share of the 98k held-out positions)", color=INK2)
a.set_ylabel("held-out loss minus master's (nats)", color=INK2); a.axhline(0, color=INK2, lw=0.8)
a.set_title("by pair frequency: the gap is in pairs seen < 10k times; long memory halves it", color=INK, fontsize=11,
            loc="left")
a.legend(loc="upper right", fontsize=9.5, frameon=False, labelcolor=INK)
a = axs[1]
# by position: the 192-sequence measurement with bootstrap errors (scripts/analysis/pos_gap.py), 4x loss_by_pos
PG = {}
for l in open(f"{D}/pos_gap_laptop.txt"):
    m = re.match(r"(gvsharp\S+)\s+(.*)$", l)
    if m:
        v = re.findall(r"([+-][\d.]+) ± ([\d.]+)", m.group(2))
        if len(v) == 5: PG[m.group(1)] = [(float(x), float(e)) for x, e in v]
PRUNS = [(r, lab, c) for r, lab, c in RUNS if r in PG] + [("gvsharp_dryspend_r1024_s0", "dry friction + spend + rank 1024", "#2b0000")]
PRUNS = [x for x in PRUNS if x[0] in PG]
npr = len(PRUNS); wp = 0.8 / npr
for i, (r, lab, c) in enumerate(PRUNS):
    xs = np.arange(5) + (i - npr / 2 + 0.5) * wp
    a.bar(xs, [v for v, _ in PG[r]], wp, color=c, yerr=[e for _, e in PG[r]], ecolor=INK2, capsize=2, label=lab)
NTOK = [192 * (b - a_) for a_, b in [(0, 2), (2, 16), (16, 128), (128, 512), (512, 2048)]]
a.set_xticks(np.arange(5)); a.set_xticklabels([f"{q}\n{n / 1000:.1f}k tok" for q, n in zip(QB, NTOK)])
a.set_xlabel("position in the 2048-token context (192 sequences; bars = ±1 bootstrap s.e.)", color=INK2)
a.axhline(0, color=INK2, lw=0.8)
a.set_ylabel("held-out loss minus master's (nats)", color=INK2)
a.set_title("by context position: runs differ by overall level, not by context length", color=INK, fontsize=11, loc="left")
a.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK)
a = axs[2]
rows = [(lab, c, POS[r][1]) for r, lab, c in RUNS if r in POS] + [("master weights", "#eb6834", POS[M][1])]
y = np.arange(len(rows))
a.barh(y, [v for *_, v in rows], color=[c for _, c, _ in rows])
a.set_yticks(y); a.set_yticklabels([lab for lab, *_ in rows], fontsize=9.5); a.invert_yaxis()
for i, (_, _, v) in enumerate(rows):
    a.annotate(f"{v:+.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
a.grid(True, color=GRID, lw=0.8, axis="x"); a.grid(False, axis="y")
a.set_xlabel("copy gain (nats): shuffled minus exact repeat", color=INK2)
a.set_title("induction (ordered copying)", color=INK, fontsize=11, loc="left")
fig.suptitle("Where the gap to master weights is, at 300M tokens (110M model; held-out loss minus master's)", color=INK,
             fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("docs/figures/gaps.png", dpi=100, facecolor=SURFACE)
print("wrote docs/figures/gaps.png", sorted(FREQ), sorted(POS))
