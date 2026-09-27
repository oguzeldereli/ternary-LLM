"""Induction on text over tokens (from checkpoints/induction_track.json): ordered copying (exact - shuffled repeat
gain) and seen-token boost (shuffled gain) per run, master as the reference.
  python -m scripts.plots.plot_induction_text   -> docs/figures/induction_text.png"""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STY = {"curve_master": ("master weights", "#eb6834"), "magadd_full": ("additive r4", "#1565c0"),
       "magadd16_qk": ("additive r16 + temperature", "#2e7d32"), "nola_lab": ("momentum, no look-ahead", "#7b1fa2"), "nola_add16": ("no look-ahead + additive r16", "#c2185b"), "nola_then_la": ("no look-ahead, then look-ahead from 131M", "#00897b"),
       "magadd16_wd_qk": ("additive r16 + WEIGHT DECAY 0.1 + temperature (running)", "#8e24aa")}
d = json.load(open("checkpoints/induction_track.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
for run, (lab, c) in STY.items():
    R = sorted((v for v in d.values() if v["run"] == run), key=lambda v: v["tokens"])
    if not R: continue
    t = [v["tokens"] / 1e6 for v in R]
    lw = 3 if run == "magadd16_wd_qk" else 2
    a.plot(t, [v["induction"] for v in R], "o-", color=c, lw=lw, label=lab)
    b.plot(t, [v["shuffled"] for v in R], "o-", color=c, lw=lw, label=lab)
a.set(xlabel="tokens (M)", ylabel="nats", title="Ordered copying (induction): exact - shuffled repeat gain")
b.set(xlabel="tokens (M)", ylabel="nats", title="Seen-token boost: gain on a shuffled repeat")
for ax in (a, b): ax.axhline(0, color="gray", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/figures/induction_text.png", dpi=130); print("wrote docs/figures/induction_text.png")
