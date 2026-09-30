"""Momentum-mechanism bench (no-look-ahead checkpoint nola_lab @131M, warm momentum, 66 steps of flips, float tail
frozen): held-out loss change and the flip signal's agreement with the true gradient at that moment, per arm.
  python -m scripts.plots.plot_mech_bench     -> docs/figures/archive/bench_mechanisms.png"""
import re, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = {}
def parse(path, suffix=""):
    gain = ""
    for l in open(path):
        m = re.match(r"== gain (\S+)", l)
        if m: gain = f" gain {m.group(1)}"
        m = re.match(r"\s+(V\d) step\s+(\d+): .*cos\(signal, truth now\) ([-+][\d.]+)\s+top-1% precision ([\d.]+)\s+held-out dL ([-+][\d.]+)", l)
        if m:
            arm = m.group(1) + (gain or suffix)
            rows.setdefault(arm, []).append((int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
parse("checkpoints/momentum_mech/momentum_mech.txt", " gain 33")
parse("checkpoints/momentum_mech/momentum_mech_gain.txt")
rows["V0 plain"] = rows.pop("V0 gain 33"); rows["V1 target point"] = rows.pop("V1 gain 33")
LAB = {"V2 gain 33": "V2 move correction, gain 33", "V3 gain 33": "V3 both, gain 33", "V2 gain 1": "V2 move correction, gain 1",
       "V3 gain 1": "V3 both, gain 1", "V2 gain 3": "V2 move correction, gain 3", "V3 gain 3": "V3 both, gain 3"}
COL = {"V0 plain": "k", "V1 target point": "#1565c0", "V2 gain 1": "#2e7d32", "V2 gain 3": "#66bb6a", "V2 gain 33": "#a5d6a7",
       "V3 gain 1": "#c62828", "V3 gain 3": "#ef5350", "V3 gain 33": "#ef9a9a"}
fig, (a, b, c) = plt.subplots(1, 3, figsize=(20, 5.6))
for arm, r in rows.items():
    r.sort(); k = [0] + [x[0] for x in r]
    a.plot(k, [0] + [x[3] for x in r], "o-", color=COL[arm], lw=2.4 if arm in ("V0 plain", "V1 target point", "V2 gain 1") else 1.5, label=LAB.get(arm, arm))
    b.plot([x[0] for x in r], [x[1] for x in r], "o-", color=COL[arm], label=LAB.get(arm, arm))
    c.plot([x[0] for x in r], [x[2] for x in r], "o-", color=COL[arm], label=LAB.get(arm, arm))
a.set(xlabel="steps of flips", ylabel="held-out loss change", title="Held-out loss change (below 0 = better)", ylim=(-0.1, 0.35))
b.set(xlabel="steps of flips", ylabel="cosine", title="Flip signal vs the true gradient at that moment")
c.set(xlabel="steps of flips", ylabel="share", title="Sign right on the signal's top 1% (0.5 = chance)")
for ax in (a, b, c): ax.grid(alpha=0.3); ax.axhline(0 if ax is not c else 0.5, color="gray", lw=0.6)
a.legend(fontsize=8)
fig.suptitle("Momentum mechanisms on the no-look-ahead bench (nola_lab @131M, warm momentum, 66 steps, float tail frozen)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/figures/archive/bench_mechanisms.png", dpi=120); print("wrote docs/figures/archive/bench_mechanisms.png")
