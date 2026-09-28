"""Induction toy, momentum mechanisms without look-ahead (seed 1 solid, seed 2 dashed): induction score over steps
and val loss. Plain momentum at the same flip rate as the control; master for reference.
  python -m scripts.plots.plot_toy_mech   -> docs/figures/toy_mechanisms.png"""
import re, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("master", None, "master (latent weights)", "#eb6834"), ("mom", "mom_s2", "plain momentum", "k"),
        ("mom_mech_v1", "mom_mech_v1_s2", "V1 target point", "#1565c0"),
        ("mom_mech_user", "mom_mech_user_s2", "your design (gain 1)", "#2e7d32"),
        ("mom_mech_user_g0", "mom_mech_user_g0_s2", "your design, no move correction", "#8e24aa")]
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2))
for s1, s2, lab, c in ARMS:
    for n, ls in ((s1, "-"), (s2, "--")):
        if n is None or not os.path.exists(f"checkpoints/toy/{n}/eval.txt"): continue
        S, I = [], []
        for l in open(f"checkpoints/toy/{n}/eval.txt"):
            S.append(int(re.search(r"step +(\d+)", l).group(1))); I.append(float(re.search(r"induction ([-+][\d.]+)", l).group(1)))
        a.plot(S, I, ls, color=c, label=lab + (" (seed 2)" if ls == "--" else ""))
        VS, VL, st = [], [], None
        for l in open(f"checkpoints/toy/{n}/train.log"):
            m = re.match(r"step +(\d+)", l)
            if m: st = int(m.group(1))
            m = re.search(r"---- val loss ([\d.]+)", l)
            if m and st is not None: VS.append(st); VL.append(float(m.group(1)))
        b.plot(VS, VL, ls, color=c, label=lab + (" (seed 2)" if ls == "--" else ""))
a.set(xlabel="step", ylabel="induction: shuffled - exact loss (nats)", title="2-layer induction toy, no look-ahead"); a.axhline(0, color="gray", lw=0.5)
b.set(xlabel="step", ylabel="val loss", title="val loss (log 256 = 5.55 is chance)")
for ax in (a, b): ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/figures/toy_mechanisms.png", dpi=120); print("wrote docs/figures/toy_mechanisms.png")
