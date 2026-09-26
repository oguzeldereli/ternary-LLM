"""One-figure summary of the project for sharing: training a ternary {-1,0,1} LLM without full-precision
master weights. (a) validation perplexity vs tokens for the progression of methods, with the optimizer
memory each needs per weight; (b) the momentum-off experiment (what the low-rank momentum does).

  python -m scripts.plots.plot_summary
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#ffffff", "#111111", "#555555", "#e6e6e6"
TOK = 32768.0


def val(d):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    v = sorted({r["step"]: r["val_loss"] for r in rows if "val_loss" in r}.items())
    return np.array([(s + 1) * TOK for s, _ in v]), np.exp(np.array([x for _, x in v]))


def train(d, k=100):
    rows = [json.loads(l) for l in open(f"checkpoints/{d}/metrics.jsonl")]
    tr = {r["step"]: r["loss"] for r in rows if "loss" in r and "tokens" in r}
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s])
    c = np.cumsum(np.insert(L, 0, 0.0))
    Ls = np.array([(c[i + 1] - c[max(0, i + 1 - min(k, i // 5 + 1))]) / min(k, i // 5 + 1) for i in range(len(L))])
    return (s + 1) * TOK, Ls


A = [  # dir, label, color, width
    ("p2_baseline", "Master weights + AdamW (standard BitNet recipe)\n  optimizer state ~16 B/weight", "#eb6834", 2.6),
    ("armA_cosine", "Stateless stochastic flips (gradient-driven)\n  0.2 B/weight (packed trits only)", "#1baf7a", 2.2),
    ("overnight_full", "+ look-ahead filter (keep a flip only if still downhill\n  on a fresh batch)  0.2 B/weight, run stopped early", "#111111", 2.2),
    ("lm_lowrank256_xb2_100M", "+ rank-256 momentum proposes the flips\n  0.2 + ~2 B/weight here; state is r(N+K), shrinks with width",
     "#d62728", 3.2),
]
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/summary.png")
a = ap.parse_args()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.25, 1]})
for d, lab, c, lw in A:
    t, p = val(d)
    ax1.plot(t, p, "o-", color=c, lw=lw, ms=4.5, label=lab)
    ax1.annotate(f"{p[-1]:.1f}", (t[-1], p[-1]), textcoords="offset points", xytext=(7, -3),
                 color=c, fontsize=11, weight="bold")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(6e6, 4.5e8); ax1.set_ylim(13, 260)
ax1.set_yticks([15, 20, 30, 50, 100, 200]); ax1.set_yticklabels(["15", "20", "30", "50", "100", "200"])
ax1.set_title("(a) Validation perplexity, 110M-parameter ternary transformer, WikiText (Llama 32k tokens)",
              loc="left", fontsize=11.5, color=INK)
ax1.set_xlabel("training tokens", color=INK2); ax1.set_ylabel("validation perplexity (log)", color=INK2)
ax1.legend(fontsize=9.3, frameon=False, loc="lower left", labelspacing=0.9)

B = [
    ("p2_baseline", "master weights + AdamW", "#eb6834", 2.0, "-"),
    ("lm_lowrank256_xb2_100M", "momentum + look-ahead", "#d62728", 3.0, "-"),
    ("r4090_la_from11M", "momentum switched OFF at 11M tokens\n(look-ahead only from the same checkpoint)", "#2a78d6", 2.6, "-"),
    ("overnight_full", "look-ahead only from scratch", "#111111", 1.8, "--"),
]
for d, lab, c, lw, ls in B:
    if not os.path.exists(f"checkpoints/{d}/metrics.jsonl"):
        continue
    t, L = train(d)
    m = t >= 3e6
    if d == "r4090_la_from11M":
        m = t >= 371 * TOK        # the 100-step mean needs a few steps after the resume
    ax2.plot(t[m], L[m], color=c, lw=lw, ls=ls, label=lab)
ax2.axvline(341 * TOK, color=INK2, lw=1, ls=":")
ax2.text(341 * TOK * 1.04, 5.6, "momentum off", color=INK2, fontsize=9.5)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(3e6, 1.2e8); ax2.set_ylim(3.2, 6.2)
ax2.set_yticks([3.5, 4, 5, 6]); ax2.set_yticklabels(["3.5", "4", "5", "6"])
ax2.set_title("(b) What the momentum does: switch it off at 11M tokens", loc="left", fontsize=11.5, color=INK)
ax2.set_xlabel("training tokens", color=INK2); ax2.set_ylabel("train loss (100-step mean, log)", color=INK2)
ax2.legend(fontsize=9.3, frameon=False, loc="lower left", labelspacing=0.9)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values(): sp.set_color("#cccccc")
fig.suptitle("Training ternary {-1, 0, +1} LLMs without full-precision master weights",
             x=0.01, ha="left", fontsize=14, weight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=150, facecolor=SURFACE)
fig.savefig(os.path.splitext(a.out)[0] + ".pdf", facecolor=SURFACE)
print("wrote", a.out)
