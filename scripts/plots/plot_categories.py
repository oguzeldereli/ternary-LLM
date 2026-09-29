"""One figure per category of what was tried. Every figure has the same two references, master weights and our
baseline (rank-256 momentum + cross-batch look-ahead x2), then only that category's runs. Left: train loss
(smoothed, log-log); middle: train loss minus the baseline's at the same tokens (below 0 = better than the
baseline); right: validation loss. A run continued on another machine (NAME_lab) is drawn from its
continuation, whose metrics include the earlier part.

  python -m scripts.plots.plot_categories      -> docs/figures/categories/*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
REFS = [("p2_baseline", "master weights", "#eb6834"),
        ("lm_lowrank256_xb2_100M", "baseline: momentum + look-ahead", "#111111")]
PAL = ["#1565c0", "#7b1fa2", "#00897b", "#fbc02d", "#c2185b", "#2e7d32", "#6d4c41", "#00acc1", "#8e24aa",
       "#ef6c00", "#546e7a"]
CATS = {
    "look_ahead": ("Look-ahead on or off (momentum both ways)", 3e8, [
        ("nola_lab", "momentum, no look-ahead"),
        ("la40_off30", "look-ahead for the first 40 steps only, then off (to 30M)"),
        ("overnight_full", "look-ahead only, no momentum (stopped at 131M)")]),
    "look_ahead_schedules": ("Look-ahead schedules", 3e8, [
        ("nola_lab", "no look-ahead throughout"),
        ("nola_then_la", "no look-ahead, then on from 131M"),
        ("la_sched", "on to 30M, off, on again from 131M"),
        ("la_sched98", "on to 30M, off, on again from 98M")]),
    "magnitude": ("Float magnitude beside the trits (additive / multiplicative low-rank)", 3e8, [
        ("magadd_full", "additive r4"),
        ("s60_magadd4", "additive r4, 60M"),
        ("magadd16_qk", "additive r16 + temperature"),
        ("magadd16_wd_qk", "additive r16 + adapter weight decay + temperature"),
        ("nola_add16", "additive r16, no look-ahead (adapter takes over q/k)"),
        ("fix4000_magadd4", "additive r4 from the step-4000 bench"),
        ("fix4000_magmul4", "multiplicative r4 from the step-4000 bench")]),
    "attention": ("Sharper attention (per-head temperature, head protection)", 3e8, [
        ("qktemp60", "per-head temperature"),
        ("qkprot60", "temperature + head protection (stopped at 28M)"),
        ("magadd16_qk", "temperature + additive r16")]),
    "early_plateau_10M": ("The no-look-ahead unigram plateau (first 10M tokens)", 1.05e7, [
        ("plateau_nola", "no look-ahead (stuck at the unigram level 2-5M)"),
        ("la40_off", "look-ahead for the first 40 steps only (1.3M), then off"),
        ("plateau_la", "look-ahead throughout (rerun with snapshots)"),
        ("nola_lab", "no look-ahead, full run")]),
    "text_read": ("Equal tokens processed: look-ahead processes 3 batches per step (x3 here); batch 48 without look-ahead the same. "
                  "The file has 398.5M tokens: 300M processed covers ~53% of it, 900M ~90%", 9.2e8, [
        ("nola_b48", "momentum, no look-ahead, batch 48 (3 x 16)"),
        ("nola_lab", "momentum, no look-ahead, batch 16"),
        ("la40_off30", "look-ahead for the first 40 steps, then off (x3 for those steps only: negligible)")]),
    "mechanisms": ("Momentum mechanisms, no look-ahead (target point, your design)", 3e8, [
        ("nola_lab", "plain rank-256 momentum (no look-ahead)"),
        ("mech_v1_b131", "V1 target point, branch at 131M"),
        ("mech_user_b131", "your design (gain 1), branch at 131M"),
        ("mech_user_g0_b131", "your design without move correction, branch at 131M"),
        ("mech_v1_s0", "V1 target point, from scratch"),
        ("mech_user_s0", "your design (gain 1), from scratch"),
        ("mech_user_g0_s0", "your design without move correction, from scratch (laptop)")]),
    "step_size": ("Step size: fewer flips per step, or accumulate then flip (no look-ahead, branches at 131M)", 3e8, [
        ("nola_lab", "plain rank-256 momentum, trained flip rate"),
        ("small_step_b131", "plain momentum at 1/4 of the flip rate"),
        ("small_step8_b131", "plain momentum at 1/8 of the flip rate (laptop)"),
        ("small_step8_lab", "plain momentum at 1/8 of the flip rate (mallard)"),
        ("accum33_b131", "accumulate 33 steps, then flip a loss-chosen subset"),
        ("mech_user_b131", "your design (gain 1), trained rate"),
        ("mech_user_q_b131", "your design (gain 1) at 1/4 of the rate"),
        ("small_step_s0", "plain momentum at 1/4 of the rate, from scratch"),
        ("small_step8_s0", "plain momentum at 1/8 of the rate, from scratch"),
        ("accum33_s0", "accumulate then flip, from scratch")]),
    "new_rules": ("Flip selection, adaptive flip rate, adaptive momentum decay (no look-ahead, branches at 131M)", 3e8, [
        ("nola_lab", "plain rank-256 momentum, trained flip rate"),
        ("small_step_b131", "plain momentum at 1/4 of the flip rate"),
        ("select_b131", "online-learned flip selector (propose 2x, keep best half)"),
        ("adaptrate_b131", "adaptive flip rate (keeps cos(g, M) near +0.03)"),
        ("multibeta_b131", "adaptive momentum decay (0.8 / 0.95 / 0.99, best predictor per layer)")]),
    "memory_and_scales": ("Per-weight memory and row/column scales (branches at 131M) vs master with its latent degraded", 3e8, [
        ("nola_lab", "plain rank-256 momentum (no look-ahead, no scales)"),
        ("rc_b131", "plain momentum + learned row/column scales"),
        ("evid3_b131", "3-bit evidence counter per weight + row/column scales"),
        ("master_q4_b131", "master, latent stored at 4 bits"),
        ("master_q3_b131", "master, latent stored at 3 bits"),
        ("master_q2_b131", "master, latent stored at 2 bits (3 levels: stateless master)"),
        ("rc_s0", "plain momentum + row/column scales, from scratch (4090)"),
        ("la40rc_s0", "+ look-ahead first 40 steps only (no plateau), from scratch (4090)"),
        ("adaptrate_rc_s0", "adaptive flip rate + row/column scales, from scratch (4090)")]),
    "speed_and_gate": ("Night 28-29 Sep: flip speed and the sign gate, from scratch with row/column scales", 3e8, [
        ("rc_s0", "plain momentum + row/column scales (4090)"),
        ("rc_s0_seed2", "the same, seed 2 (seed spread)"),
        ("speedref_rc_s0", "speed reference: divide M by a slow EMA of its size"),
        ("gate_rc_s0", "sign gate: flip only where this batch agrees with M"),
        ("gate_rc_s0_seed2", "sign gate, seed 2"),
        ("gatevnorm_rc_s0", "sign gate + factored Adam step"),
        ("vnorm_rc_s0", "factored Adam step (smaller steps where steep)"),
        ("rate085_rc_s0", "rate control for the gate: plain at 0.85x flip rate"),
        ("rate064_rc_s0", "rate control for gate + Adam step: plain at 0.64x flip rate"),
        ("gatevnorm_rc_s0_seed2", "sign gate + factored Adam step, seed 2"),
        ("vnorm064_rc_s0", "factored Adam step at 0.64x flip rate (no gate)")]),
    "momentum_fixes_20M": ("Momentum fixes, 20M from-scratch screens", 2.2e7, [
        ("s20_base", "baseline seed 0"), ("s20_base_seed1", "baseline seed 1"), ("s20_base_seed2", "baseline seed 2"),
        ("s20_gate", "sign gate"), ("s20_vnorm99", "vnorm"), ("s20_gate_vnorm99", "gate + vnorm"),
        ("s20_spend2", "spend"), ("s20_refresh16", "refresh"), ("s20_maskstuck", "stuck mask"),
        ("s20_magadd4", "additive r4"), ("s20_magmul4", "multiplicative r4")]),
}

DLIM = {"early_plateau_10M": (-0.2, 1.4)}   # difference-panel range where the default +-0.45 clips


def load(d):
    if os.path.exists(f"checkpoints/{d}_lab/metrics.jsonl"): d = f"{d}_lab"
    f = f"checkpoints/{d}/metrics.jsonl"
    if not os.path.exists(f): return None
    tr, va, tps = {}, {}, 32768.0
    for l in open(f):
        r = json.loads(l)
        if "loss" in r and "tokens" in r:
            tr[r["step"]] = r["loss"]; tps = r["tokens"] / (r["step"] + 1)   # tokens per step (batch size varies)
        if "val_loss" in r: va[r["step"]] = r["val_loss"]
    if len(tr) < 10: return None
    s = np.array(sorted(tr)); L = np.array([tr[i] for i in s]); t = (s + 1) * tps
    vs = np.array(sorted(va)); V = np.array([va[i] for i in vs])
    return t, L, (vs + 1) * tps, V


def smooth(L, k=100):
    c = np.cumsum(np.insert(L, 0, 0.0))
    w = [min(k, i // 5 + 1) for i in range(len(L))]
    return np.array([(c[i + 1] - c[i + 1 - w[i]]) / w[i] for i in range(len(L))])


os.makedirs("docs/figures/categories", exist_ok=True)
XMULT = {"text_read": {"lm_lowrank256_xb2_100M": 3.0}}   # per figure: tokens read per logged token


for key, (title, tmax, runs) in CATS.items():
    fig, (a, dax, b) = plt.subplots(1, 3, figsize=(24, 6.8), facecolor=SURFACE)
    tb, Lb, *_ = load("lm_lowrank256_xb2_100M"); Lb = smooth(Lb)
    tb = tb * XMULT.get(key, {}).get("lm_lowrank256_xb2_100M", 1.0)
    base = lambda t: np.interp(np.log(t), np.log(tb), Lb)
    lines = [(d, lab, c, 2.2, "--") for d, lab, c in REFS] + \
            [(d, lab, PAL[i % len(PAL)], 2.4, "-") for i, (d, lab) in enumerate(runs)]
    for d, lab, c, lw, ls in lines:
        r = load(d)
        if r is None: continue
        t, L, vt, V = r
        xm = XMULT.get(key, {}).get(d, 1.0); t, vt = t * xm, vt * xm
        m = t <= tmax * 1.05
        a.plot(t[m], smooth(L)[m], ls, color=c, lw=lw, label=lab)
        md = m & (t >= 3e5) & (t <= tb.max())
        dax.plot(t[md], smooth(L)[md] - base(t[md]), ls, color=c, lw=lw, label=lab)
        mv = vt <= tmax * 1.05
        if mv.any(): b.plot(vt[mv], V[mv], "o" + ls, color=c, lw=lw, ms=3.5, label=lab)
    xl = ("tokens processed (random windows with repeats; look-ahead counts its check batches)" if key in XMULT
          else "tokens")
    a.set(xscale="log", yscale="log", xlim=(3e4, tmax * 1.05), xlabel=xl, ylabel="train loss (smoothed)")
    b.set(xscale="log", xlim=(3e6 if tmax > 1e8 else 1e6, tmax * 1.05), xlabel=xl, ylabel="validation loss")
    vals = [x for d, *_ in lines if (r := load(d)) is not None
            for x in r[3][r[2] * XMULT.get(key, {}).get(d, 1.0) <= tmax * 1.05]]
    if vals: b.set_ylim(min(vals) - 0.05, min(max(vals), min(vals) + 2.0))
    dax.set(xscale="log", xlim=(3e5, tmax * 1.05), xlabel=xl, ylabel="train loss minus baseline",
            ylim=DLIM.get(key, (-0.45, 0.45))); dax.axhline(0, color=INK2, lw=0.8)
    for ax in (a, dax, b):
        ax.set_facecolor(SURFACE); ax.grid(True, which="both", color=GRID, lw=0.6)
        for sp in ax.spines.values(): sp.set_color(GRID)
    a.legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"docs/figures/categories/{key}.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", f"docs/figures/categories/{key}.png")
