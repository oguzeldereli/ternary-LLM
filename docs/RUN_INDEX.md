# Run index

Running, queued, unfinished and never-run runs: [QUEUE.md](QUEUE.md) (kept current). Formulas of every flag:
[FORMULAS.md](FORMULAS.md). Figure of the best runs: [figures/best_runs.png](figures/best_runs.png).

## Best so far (110M, 300M tokens, from scratch, final validation loss / perplexity)

| run | what | val loss | ppl |
|---|---|---|---|
| `fp32_baseline` | full precision, fp32 + AdamW (reference) | 2.683 | 14.6 |
| `master_tracked` | master weights: fp32 latent + STE + AdamW (reference) | 2.751 | 15.7 |
| **`gvsharp_dry_r1024_s0`** | sharp base + beta 1 + dry friction 1/33 + rank 1024 | **2.8198** | **16.8** |
| **`gvsharp_dryspend_r512_s0`** | sharp base + beta 1 + dry friction 1/33 + spend 3 + rank 512 | **2.8215** | **16.8** |
| `gvsharp_b0995_r512_s0` | sharp base + decay 0.995 + rank 512 | 2.8339 | 17.0 |
| `gvsharp_dry_r512_s0` | sharp base + dry friction + rank 512 | 2.8356 | 17.0 |
| `gvsharp_dryspend_s0` | sharp base + dry friction + spend (rank 256) | 2.8699 | 17.6 |
| `gvsharp_rc_s0` | sharp base: gate + Adam step + row/col scales + additive r16 + adapter wd + head temperature | 2.9875 | 19.8 |

The sections below keep each phase's own unit: the first phases report perplexity, from 27 Sep on validation loss.

Every directory under `checkpoints/` (gitignored). Unless noted: 110M (`small`), wiki32k,
seq 2048, 16 x 2048 = 32,768 tokens/step, same seed and batch order, kernel mode with
int8 forward. "val" = final validation perplexity. Details and discussion: [RUNS.md](RUNS.md).
Each dir has `metrics.jsonl` (+ `ckpt.pt`); logs are `train.log` in the dir, or
`checkpoints/<name>.log` for older runs.

## Ceiling

| run | what | tokens | val |
|---|---|---|---|
| `p2_baseline` | **master weights** (fp32 latent + STE + AdamW, LR 1.5e-3) | 300M | **15.8** |
| `probe_master` | master, 161 steps with probes (`--probe`) | 5M | 326.9 |
| `master_tracked` | master weights, the reference of every later section (val loss 2.751) | 300M | 15.7 |
| `curve_master` | master with window-curve tracking and snapshots every 1000 steps (stopped at 6350) | 208M | - |
| `fp32_baseline` | full precision fp32 + AdamW (val loss 2.683) | 300M | 14.6 |

## Stateless flips, full length (all bf16 float tail = frozen norm gains)

| run | what | tokens | val |
|---|---|---|---|
| `armA_cosine` | **reference**: flip rate 0.02 cosine-annealed to 0 | 300M | **98.6** |
| `armA_cos_600M` | reference, 600M-token schedule | 600M | 92.7 |
| `armA_cos_r005` | cosine from 4x lower rate (0.005) | 300M | 101.0 |
| `armA_linear` | linear anneal | 300M | 101.4 |
| `armA_cos_lockout` | cosine + per-weight flip lockout | 300M | 102.7 |
| `armA_cos_floor` | cosine to a 0.02% floor | 300M | 104.0 |
| `p3b_acc1` | constant rate | 300M | 135.8 |
| `p3b_acc4` | constant rate, 4x batch | 400M | 135.5 |
| `armA_cos_ef` | cosine + spatial error feedback | 300M | 156.0 |
| `armB_absscale` | frozen absolute threshold | 300M | 160.7 |
| **`armA_cos_la1`** | **look-ahead filter** (`--lookahead 1`), reference schedule; stopped at step 6171, resumable | 202M | **65.3** |

## Short probes and screens (1000 steps or fewer)

| run | what | tokens | val |
|---|---|---|---|
| `lr_ctl` | control for the 1000-step probes | 33M | 156.8 |
| `armA_cos_lr15` | 1.5x float-tail LR | 34M | 153.3 |
| `lo_once_t200`, `lo_once_t1000`, `lo_norev_t1000` | lockout variants | 33M | 175-191 |
| `ef_probe_a0*` | error-feedback strength probes | 20M | 256-828 |
| `gref1/7/10/25/100`, `gref10_r3x` | g_ref sweep (300-step alpha screens) | 10M | 350-490 |
| `rownorm`, `invmag`, `hump` | flip-rule shape screens | 10M | 350-431 |
| `ev3_screen` | 3-bit evidence counter (not stateless) | 10M | 554 |
| `rs_smoke` | greedy flip-rate search | 5M | 633 |
| `la1`, `la2` | look-ahead 1 and 2 passes, alpha screens | 10M | 280, 254 |
| `la_rwarm` | look-ahead, rate warmup 0.02 -> 0.04 (stopped at 790; snapshots every 100) | 26M | - |
| `la_r0ramp` | look-ahead, rate ramp 0 -> 0.02 over 305 steps (stopped at 313) | 10M | - |
| `la_fastramp_lr15` | look-ahead, rate ramp over 30 steps + master's tail LR (snapshots 100/200/300) | 10M | 180.7 |
| `la_fast_fp32tail` | same + **fp32 float tail** (`--tail_fp32`), probes | 10M | 179.2 |
| `la_fast_fp32tail_r04` | same, peak rate 0.04 | 10M | (running) |
| `probe_fast` | `la_fastramp_lr15` config, 161 steps with probes | 5M | 317.3 |

## Scratch / smoke tests (safe to delete)

`acc_a`, `acc_b`, `p2_master`, `sanity_master`, `smoke`, `t_armA`, `t_armB`,
`la_r0ramp_lr15` (stopped at step 20), `p3b_acc16`, `p3b_acc64` (never ran),
`wiki_ev`, `wiki_ev3`, `wiki_mf` (early 51M/GPT-2-vocab runs from RESULTS.md, checkpoint only).


## Low-rank momentum and cross-batch look-ahead (24-27 Sep)

Screens at 10M tokens report val loss; the MLP testbed is a ternary MLP language model (`scripts/mlp/mlp_lab.py`).

| run | what | tokens | val (loss) |
|---|---|---|---|
| `lm_frozen` | ternary layers frozen at their random init (flip rate 0) | 10M | 5.698 |
| `lm_plain` | stateless flips from the gradient | 10M | 5.684 |
| `la_xb2` | cross-batch look-ahead x2 (proposals from g) | 10M | 5.002 |
| `la_xb2_rc` | + row/column scales | 10M | 4.974 |
| `lm_lowrank256` | rank-256 momentum, no look-ahead (stalls at the unigram level) | 10M | 6.047 |
| `lm_lowrank256_r04` / `_adapt` | rate 0.04 / angle-scaled decay | 10M | 6.072 / 6.293 |
| `lm_lowrank256_xb2` | rank-256 momentum proposes, look-ahead x2 keeps | 10M | 4.892 |
| **`lm_lowrank256_xb2_100M`** | the same, full 300M schedule | 300M | **3.127** |
| `overnight_full` | `la_xb2` full schedule (stopped) | 130M | - |
| `magadd_full` | look-ahead + momentum + additive r4 float term | 300M | 3.082 |
| `mlp/mlp_master` / `mlp_lowrank64` / `mlp_lowrank256` / `mlp_sketch64` / `mlp_frozen` | MLP testbed (`checkpoints/mlp/`): master / rank-64 momentum / rank 256 / count sketch / frozen | 6000 steps | 4.351 / 4.670 / 4.577 / 4.804 / 5.066 |

## No look-ahead, momentum mechanisms, step size (27-28 Sep)

All no look-ahead unless noted; "branch" = continued from `nola_lab` at step 4000 (131M) with its momentum.
val = final validation loss (not perplexity) at the given tokens. Details: [RUNS.md](RUNS.md), status of running
ones: [QUEUE.md](QUEUE.md).

| run | what | tokens | val |
|---|---|---|---|
| `nola_lab` | rank-256 momentum, no look-ahead (4090 to 30M, lab 3090 to 300M) | 300M | 3.158 |
| `nola_then_la` | `nola_lab` to 131M, then look-ahead x2 on | 300M | 3.1225 |
| `la_sched` | look-ahead 0-30M, off to 131M, on after | 300M | 3.1108 |
| `la_sched98` | look-ahead 0-30M, off to 98M, on after | 300M | 3.1135 |
| `la40_off30` | look-ahead the first 40 steps, then off | 30M | 4.506 |
| `nola_b48` | no look-ahead, batch 48 (3 x 16: the text a look-ahead step reads) | 9155 steps (900M read) | 2.889 |
| `nola_add16` | no look-ahead + additive r16 (stopped: the adapter takes over q/k) | 112M | ~4.6 |
| `magadd16_qk_lab` | look-ahead + additive r16 + per-head temperature (to its planned 200M) | 200M | 3.1212 |
| **`magadd16_wd_qk_lab`** | look-ahead + additive r16 + adapter weight decay 0.1 + per-head temperature | 300M | **2.9925** |
| `mech_v1_b131` | V1 target point (`--mech v1`), branch | 300M | 3.1215 |
| `mech_user_b131` | your design (`--mech user`, gain 1), branch | 300M | 3.1202 |
| `mech_user_g0_b131` | your design without the move correction (gain 0), branch | 300M | 3.1325 |
| `mech_v1_s0` | V1 from scratch | 131M | 3.582 |
| `mech_user_s0` | your design (gain 1) from scratch | 131M | 3.505 |
| `mech_user_g0_s0` | your design (gain 0) from scratch | 131M | 3.548 |
| `small_step_b131` | 1/4 of the flip rate, branch | 300M | 3.1551 |
| `small_step_s0` | 1/4 of the flip rate, from scratch | 300M | 3.1933 |
| `accum33_b131` | accumulate 33 steps without flips, then flip a loss-chosen subset, branch | 300M | 3.2795 |
| `adaptrate_b131` | adaptive flip rate (`--adapt_rate`, keeps cos(g, M) near 0.03), branch | 300M | 3.1604 |
| `multibeta_b131` | adaptive momentum decay (`--multibeta 0.8,0.95,0.99`), branch | 300M | 3.1926 |
| `select_b131` | online flip selector (`--select`), branch; stopped | 229M | 3.366 |
| `rc_b131` | plain momentum + learned row/column scales (`--rc_scale`), branch | 300M | 3.1480 |
| `evid3_b131` | 3-bit evidence counter per weight (`--evidence_bits 3`) + row/column scales, branch (not stateless) | 300M | 3.0970 |
| `master_q4_b131` | master, latent stored at 4 bits (`--master_bits 4`), branch of `master_tracked` at 131M | 300M | 2.9231 |
| `master_q3_b131` | master, latent at 3 bits | 300M | 3.1222 |
| `master_q2_b131` | master, latent at 2 bits (3 levels: a stateless master) | 300M | 3.4686 |
| `rc_s0` | plain momentum + learned row/column scales, from scratch (4090) | 300M | 3.1307 |
| `rc_s0_seed2` | the same, seed 2 | 300M | 3.1273 |
| `speedref_rc_s0` | + per-layer slow speed reference (`--speed_ref 0.995`) | 300M | 3.1315 |
| `gate_rc_s0` / `gate_rc_s0_seed2` | + sign gate (`--lr_gate`, now also without look-ahead) | 300M | 3.1194 / 3.1138 |
| `rate085_rc_s0` | plain at 0.85x peak flip rate (the gate's rate control) | 300M | 3.1208 |
| `vnorm_rc_s0` | + factored Adam step (`--lr_vnorm 0.99`) | 300M | 3.1063 |
| **`gatevnorm_rc_s0`** / `gatevnorm_rc_s0_seed2` | + gate + Adam step | 300M | **3.0773 / 3.0699** |
| `rate064_rc_s0` | plain at 0.64x peak flip rate (gate + Adam step's rate control) | 300M | 3.1290 |
| `vnorm064_rc_s0` | Adam step at 0.64x peak rate, no gate | 300M | 3.1224 |
| `user_cap3_s0` | user's rule: `--lr_beta 1 --mom_ncap 3 --pfun_tanh 10` | 300M | 3.2168 |
| `user_cap1_s0` | user's rule, cap 1 (stopped: too few flips, loses no-context statistics) | ~115M | ~3.83 |
| `user_cap1_flip15_s0` | user's rule, cap 1, flip scale 1.5 (stopped) | 197M | 3.635 |
| `usersweep_cap{1,2,3,10,30}` | user's rule, cap sweep, stop at step 1500 | 49M | 4.132 / - / 3.991 / 4.946 / 5.305 |
| `grav05_rc_s0` | asymmetric gravity 0.5 (`--grav_up 0.5`) | 300M | 3.5014 |
| `grav05_gatevnorm_rc_s0` | asymmetric gravity 0.5 + gate + Adam step | 300M | 3.4972 |
| `gvundo_rc_s0` | gate + Adam step + undo (`--undo`); stopped (undo reverses 2% of flips: inert) | 66M | 3.585 |
| **`gvsharp_rc_s0`** | gate + Adam step + sharp (`--lowrank_mag add:16 --mag_wd 0.1 --qk_temp`): best without look-ahead, beats look-ahead + sharp (2.992) | 300M | **2.9875** |
| `gvsharp_b099_s0` | sharp base + `--lr_beta 0.99` | 300M | 2.9108 |
| `gvsharp_b1_s0` | sharp base + `--lr_beta 1` (relative flip rule, no cap) | 300M | 3.0466 |
| `gvsharp_r512_s0` | sharp base + `--lowrank 512` | 300M | 2.9627 |
| `gvsharp_slowgate_s0` | sharp base + `--slow_gate 64 --slow_beta 0.999` | 300M | 2.9615 |
| `gvsharp_dither_s0` | sharp base + `--dither_ld` (low-discrepancy flip draw) | 300M | 2.9908 |
| **`gvsharp_dry_s0`** | sharp base + `--lr_beta 1 --dry_vec 0.0303` | 300M | **2.8812** |
| `gvsharp_b1spend_s0` | sharp base + `--lr_beta 1 --spend 3` | 300M | 2.9426 |
| `gvsharp_b1spend_slowgate_s0` | sharp base + beta 1 + spend 3 + slow gate | 300M | 2.9708 |
| `gvsharp_b0995_s0` | sharp base + `--lr_beta 0.995` | 300M | 2.8783 |
| `gvsharp_dry01_s0` | sharp base + beta 1 + `--dry_vec 0.01` | 300M | 2.9558 |
| `gvsharp_dry1_s0` | sharp base + beta 1 + `--dry_vec 0.1` | 300M | 3.0031 |
| **`gvsharp_dryspend_s0`** | sharp base + beta 1 + dry 0.0303 + spend 3 | 300M | **2.8699** |
| `gvsharp_b099spend_s0` | sharp base + beta 0.99 + spend 3 | 300M | 2.9031 |
| `gvsharp_dryw_s0` | sharp base + beta 1 + `--dry_w 0.0303` (per weight) | 300M | 2.9409 |
| `gvsharp_b099slow_s0` | sharp base + beta 0.99 + slow gate | 300M | 2.9441 |
| **`gvsharp_dry_r512_s0`** | sharp base + beta 1 + dry 0.0303 + rank 512 | 300M | **2.8356** |
| `gvsharp_b0995spend_r512_s0` | sharp base + beta 0.995 + spend 3 + rank 512 | running (bufflehead) | - |
| **`gvsharp_dryspend_r512_s0`** | sharp base + beta 1 + dry 0.0303 + spend 3 + rank 512 | 300M | **2.8215** |
| `gvsharp_b0995_r512_s0` | sharp base + beta 0.995 + rank 512 | 300M | 2.8339 |
| `gvsharp_b0998_s0` | sharp base + beta 0.998 | 300M | 2.8910 |
| **`gvsharp_dry_r1024_s0`** | sharp base + beta 1 + dry 0.0303 + rank 1024 | 300M | **2.8198** |
| `gvsharp_b0995spend_s0` | sharp base + beta 0.995 + spend 3 | 300M | 2.8807 |
| `gvsharp_dry05_s0` | sharp base + beta 1 + dry 0.05 | 300M | 2.8875 |
| `gvsharp_dry02_s0` | sharp base + beta 1 + dry 0.02 | 300M | 2.8970 |

## Rank sweep and model width (30 Sep, running)

Recipe = sharp base + beta 1 + dry 0.0303 + spend 3. Does it survive a rank that fits 27B in 12-16 GB, and does the
rank needed grow with width? 340M = `--preset d1024_l24`.

| run | what | tokens | val |
|---|---|---|---|
| `rk32_dryspend_s0` / `rk64_dryspend_s0` / `rk128_dryspend_s0` | 110M, rank 32 / 64 / 128 | running | - |
| `big_rk64_dryspend_s0` / `big_rk128_dryspend_s0` | 340M, rank 64 / 128 (lab) | running | - |
| `big_dryspend_r512_s0` | 340M, rank 512 (4090) | running | - |
| `big_master` | 340M master weights (4090, after the above) | queued | - |
