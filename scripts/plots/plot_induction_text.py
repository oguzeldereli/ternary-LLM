"""Induction on text vs validation loss, per run (from checkpoints/induction_track.json and metrics.jsonl):
val loss, ordered copying (exact - shuffled repeat gain) and seen-token boost (shuffled gain) over tokens.
  python -m scripts.plots.plot_induction_text   -> docs/figures/induction_text.png"""
import json, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STY = {  # run: (label, color, width)
    "curve_master": ("master weights", "#eb6834", 2.2),
    "lm_lowrank256_xb2_100M": ("momentum + look-ahead (baseline)", "#e53935", 1.6),
    "magadd_full": ("momentum + look-ahead + additive r4", "#1565c0", 1.6),
    "magadd16_wd_qk": ("look-ahead + additive r16 + wd + temperature", "#8e8c85", 1.6),
    "nola_lab": ("MOMENTUM, NO LOOK-AHEAD", "#7b1fa2", 3.0),
    "nola_then_la": ("NO LOOK-AHEAD, THEN LOOK-AHEAD FROM 131M", "#00897b", 3.0),
    "nola_add16": ("no look-ahead + additive r16", "#c2185b", 2.0),
}
d = json.load(open("checkpoints/induction_track.json"))
fig, (v, a, b) = plt.subplots(1, 3, figsize=(20, 5.6))
for run, (lab, c, lw) in STY.items():
    f = f"checkpoints/{run}/metrics.jsonl"
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f)]
        V = sorted({r["step"]: r["val_loss"] for r in rows if "val_loss" in r}.items())
        if V: v.plot([(s + 1) * 32768 / 1e6 for s, _ in V], [x for _, x in V], "-", color=c, lw=lw, label=lab)
    R = sorted((x for x in d.values() if x["run"] == run), key=lambda x: x["tokens"])
    if run == "nola_then_la":        # the branch starts at nola_lab's step 4000
        R = [x for x in d.values() if x["run"] == "nola_lab" and x["step"] == 4000] + R
    if not R: continue
    t = [x["tokens"] / 1e6 for x in R]
    a.plot(t, [x["induction"] for x in R], "o-", color=c, lw=lw, ms=4, label=lab)
    b.plot(t, [x["shuffled"] for x in R], "o-", color=c, lw=lw, ms=4, label=lab)
v.set(xlabel="tokens (M)", ylabel="val loss", title="Validation loss", ylim=(2.8, 4.2), xlim=(30, 310))
a.set(xlabel="tokens (M)", ylabel="nats", title="Ordered copying (induction): exact - shuffled repeat gain")
b.set(xlabel="tokens (M)", ylabel="nats", title="Seen-token boost: gain on a shuffled repeat")
for ax in (v, a, b): ax.grid(alpha=0.3)
for ax in (a, b): ax.axhline(0, color="gray", lw=0.5)
v.legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/figures/induction_text.png", dpi=120); print("wrote docs/figures/induction_text.png")
