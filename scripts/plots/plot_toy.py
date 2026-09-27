"""Induction toy: induction score (shuffled - exact repeat loss) and val loss vs step for every arm.
  python -m scripts.plots.plot_toy   -> docs/figures/toy_induction.png"""
import re, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("master", "master (latent weights)", "k"), ("mom_la", "momentum + look-ahead", "C0"),
        ("mom_add_la", "momentum + additive + look-ahead", "C1"), ("mom_r005", "momentum, rate 0.005", "C2"),
        ("mom", "momentum, rate 0.02", "C3"), ("mom_add", "momentum + additive, rate 0.02", "C4")]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
for n, lab, c in ARMS:
    S, I = [], []
    for l in open(f"checkpoints/toy/{n}/eval.txt"):
        S.append(int(re.search(r"step +(\d+)", l).group(1))); I.append(float(re.search(r"induction ([-+][\d.]+)", l).group(1)))
    a.plot(S, I, color=c, label=lab)
    VS, VL, st = [], [], None
    for l in open(f"checkpoints/toy/{n}/train.log"):
        m = re.match(r"step +(\d+)", l)
        if m: st = int(m.group(1))
        m = re.search(r"---- val loss ([\d.]+)", l)
        if m and st is not None: VS.append(st); VL.append(float(m.group(1)))
    b.plot(VS, VL, color=c, label=lab)
a.set(xlabel="step (32 x 256 tokens)", ylabel="induction: shuffled - exact loss (nats)",
      title="2-layer toy, repeated random segments (vocab 256)"); a.axhline(0, color="gray", lw=0.5)
b.set(xlabel="step", ylabel="val loss", title="val loss (log 256 = 5.55 is chance)")
a.legend(fontsize=8); b.legend(fontsize=8)
for ax in (a, b): ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("docs/figures/toy_induction.png", dpi=130)
print("saved docs/figures/toy_induction.png")
