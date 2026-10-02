# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16). From
2026-09-27 on, new runs are without look-ahead.

Last updated: 2026-10-02 23:25

Run names: `rc` = learned row/column scales, `s0` = from scratch (step 0), `b131` = branch from `nola_lab` at 131M,
`gate` = sign gate (flip only where this batch's gradient agrees with the momentum), `vnorm` = factored Adam step
(momentum divided by a row x column gradient size), `rateNNN` = plain momentum at 0.NNx peak flip rate,
`speedref` = per-layer slow speed reference, `seed2` = same run with seed 2.

## Now (2 Oct, 23:25)

Storage-format runs (`--ts` ranks 512, fp32 twin 2.7067; the fp32 rerun with diagnostics is -0.001, so noise is about
+-0.005), gap to the fp32 twin at step 8000 (7750 where marked):

| machine | run | storage | gap | status |
|---|---|---|---|---|
| 4090 | `qabf_s0` | accumulator bf16 | +0.004 | **done, 2.7105** |
| harlequin / pintail | `q8a_s0` / `q8noV_s0` | accumulator int8 / int8 with V fp32 | +0.012 (9000) / +0.009 (8750) | ~23:35 |
| flounder / barbel | `q8m_s0` / `qmf8_s0` | direction int8 / fp8 | +0.004 / -0.001 (7750) | ~00:00 |
| koi / quillback | `qa16_s0` / `qbf16_s0` | accumulator fp16 (7750) / bf16 everywhere | +0.002 / +0.002 | ~00:00 |
| albacore / elver / hake | `qm8a16_s0` / `qmf8a16_s0` / `qm8abf_s0` | direction int8 / fp8 / int8 + accumulator fp16 / fp16 / bf16 (7750) | +0.002 / -0.003 / +0.002 | ~00:00 |
| rudd | `qfp16_s0` | fp16 everywhere | +0.012 | ~00:00 |
| goldeye | `qm16a8_s0` | direction fp16 + accumulator int8 (7750) | +0.023 | ~00:00 |
| dory / uaru / inanga | `q8all_s0` / `q8det_s0` / `q8row_s0` | int8 everywhere / nearest rounding / per row | +0.023 / +0.017 / +0.054 | ~00:00 |
| tope | `qfp8_s0` | fp8 everywhere | +0.251 | ~00:00 |
| lamprey | `qdiag_fp32_s0` | fp32 with diagnostics | -0.001 | ~00:00 |

Myriad: 40267 `x1b_ts512_la1000an05` at 3460 (-0.058 vs 1.3B master at 3000; 9.7 s/step), Sat ~14:40;
36472 `x1b_dryspend_r512_lr75` at 8380 (+0.138 at 8000), Sat ~01:15; 43340-43342 (`prof_tsfast`, `prof_tsfastf`,
`prof_oldfast`) waiting in the queue since 21:18. The 4090 is idle.

## Earlier (2 Oct, 18:45): why int8 costs (storage-format ablation)

`--ts` ranks 512, steps x ratio^0.5, tau 1000, fused (fp32 reference 2.7067 / 2.6978), one storage change each:

| machine | run | storage of the low-rank matrices |
|---|---|---|
| dory | `q8all_s0` | int8 (per column, stochastic) everywhere |
| flounder / harlequin | `q8m_s0` / `q8a_s0` | int8 on the short momentum only / the accumulator only |
| pintail | `q8noV_s0` | int8, the orthonormal bases V kept fp32 |
| uaru / inanga | `q8det_s0` / `q8row_s0` | int8 rounded to nearest / scaled per row |
| quillback / rudd / tope | `qbf16_s0` / `qfp16_s0` / `qfp8_s0` | bf16 (8 exp + 7 mantissa) / fp16 (5 + 10) / fp8 e4m3 with a per-column scale |
| albacore / elver | `qm8a16_s0` / `qmf8a16_s0` | direction int8 / fp8 + accumulator fp16 (`--ts_qfmt_m`, `--ts_qfmt_a`) |
| hake / goldeye | `qm8abf_s0` / `qm16a8_s0` | direction int8 + accumulator bf16 / the reverse: direction fp16 + accumulator int8 |
| barbel / koi / 4090 | `qmf8_s0` / `qa16_s0` / `qabf_s0` | direction fp8 alone / accumulator fp16 alone / accumulator bf16 alone |
| lamprey | `qdiag_fp32_s0` | fp32, logging every 250 steps: crest factor, share of per-step changes below half an int8 step, change / int8 step, log2 dynamic range |

All ~23:45. Myriad: `x1b_ts512_la1000an05` (Sat ~14:45), `x1b_dryspend_r512_lr75` (Sat ~01:20), profile jobs queued.

## Earlier (2 Oct, 17:20)

| machine | run | what | expected |
|---|---|---|---|
| Myriad 40267 | `x1b_ts512_la1000an05` | 1.3B `--ts` ranks 512: +0.011 vs 1.3B master at 1500 (old rule was +0.93 at 1000); 9.7 s/step, 24.7 GiB | Sat ~14:45 |
| Myriad 36472 | `x1b_dryspend_r512_lr75` | 1.3B old rule: +0.187 vs 1.3B master at 6000 | Sat ~01:20 |
| Myriad 40272-40274 | `prof1b_*` | 1.3B A100 profile (master / old / --ts) | queued |
| plaice / hake | `ts16rs512la1000an05int8_s0` / `ts16r256rs256la1000an05int8_s0` | int8 + fused: ~0.03 behind their fp32 twins | ~18:15 |
| albacore / elver | `ts16rs512la1000an05_s0` / `ts16r256rs256la1000an05_s0` | both late fixes (fp32 twins) | ~17:45 |
| tope / lamprey | `mx_extras_s0` / `_seed2` | master + extras (fair reference), at 9000 | ~17:30 |
| goldeye / barbel | `ts16fullla1000_seed2` / `ts16rs512la1000_seed2` | seed 2 of the annealed-leak runs | ~17:45 |

Finished today: 340M `--ts` 2.6268; seed 2s 2.6865 (full) / 2.6978 (rank 512, ^0.5); ^0.5 full 2.6804; ^0.75 rank 512
2.7138; ranks 256 ^0.5 2.7824. Free: 4090, dory, flounder, harlequin, pintail, uaru, inanga, quillback, rudd, koi.

## Earlier (2 Oct, 12:20)

| machine | run | what | expected |
|---|---|---|---|
| tope / lamprey | `mx_extras_s0` / `mx_extras_seed2` | **fair reference**: master + the same float extras as our runs (row/col scales, additive rank-16 adapter, qk temperature) | ~16:30 |
| dory / goldeye | `ts16full_tau1000_seed2` / `ts16fullla1000_seed2` | seed 2 of the full-rank `--ts` runs (2.6982 / 2.6746) | ~16:00-17:00 |
| flounder / barbel | `ts16rs512tau1000an05_seed2` / `ts16rs512la1000_seed2` | seed 2 of the rank-512 `--ts` runs (2.7067 / 2.7115) | ~16:00-17:00 |
| harlequin / pintail / uaru | `ts16fulltau1000an05_s0` / `ts16rs512tau1000an075_s0` / `ts16r256rs256tau1000an05_s0` | annealing variants | ~13:30 |
| skate | `ts16r128rs512tau1000_s0` | accumulator 128 + short 512: lost (skate off the network at ~13:20, +0.026 at 6000) | - |
| plaice / hake | `ts16rs512la1000an05int8_s0` / `ts16r256rs256la1000an05int8_s0` | the same with both low-rank matrices on an int8 grid (`--ts_int8`) and the update inside the backward (`--ts_fused`); 200-step test: fused = plain (5.678 vs 5.681), int8 +0.02 | ~18:15 |
| albacore / elver | `ts16rs512la1000an05_s0` / `ts16r256rs256la1000an05_s0` | `--ts` ranks 512 / 256 with both late-phase fixes: leak and steps scaled by (lr ratio)^0.5. The fixed-leak runs end at a third of master's slope (0.010 vs 0.029 per 1000 steps), the scaled leak 0.020, steps x ratio^0.5 0.034 | ~18:00 |
| 4090 | `big_ts16rs512tau1000_s0` | `--ts` 512 / 512 at 340M: -0.138 vs 340M master at 4500 | ~18:00 |
| Myriad 36472 | `x1b_dryspend_r512_lr75` | 1.3B old rule at lr 7.5e-4 (1.3B master 2.5868): +0.208 at step 4000, 9.2 s/step | Sat ~01:30 |
| Myriad 40272-40274 (queued) | `prof1b_master` / `prof1b_old` / `prof1b_ts` (`profile_1b_one.sh`, 15 min each) | 1.3B on one A100, 25 steps: step timer + memory every step, torch profile of 5 steps (kernels + trace), nvidia-smi every second: why our step is 9.2 s vs master's 4.7 s | ~15 min after start |
| Myriad 40267 (queued) | `x1b_ts512_la1000an05` | 1.3B `--ts`, ranks 512, tau 1000, leak and steps x (lr ratio)^0.5, lr 7.5e-4 (`run_1b_ts.sh`) | ~1-1.5 days after start |

Free lab121: albacore, elver, hake, inanga, koi (another user's Ray worker), plaice, quillback, rudd.

## Earlier (2 Oct, 10:55)

| machine | run | what | expected |
|---|---|---|---|
| dory / flounder | `ts16full_tau1000_seed2` / `ts16rs512tau1000an05_seed2` | seed 2 of the two headline `--ts` runs (2.6982 / 2.7067) | ~15:45 |
| elver / quillback / inanga / albacore | `ts16fullla1000_s0` / `ts16rs1024la1000_s0` / `ts16rs512la1000gate_s0` / `ts16rs512la300_s0` | annealed-leak `--ts` runs, at 8500-9000 | ~11:15 |
| harlequin / pintail / uaru | `ts16fulltau1000an05_s0` / `ts16rs512tau1000an075_s0` / `ts16r256rs256tau1000an05_s0` | annealing ^0.5 full rank / ^0.75 rank 512 / ^0.5 ranks 256 | ~13:30 |
| lamprey / goldeye / tope / koi / rudd / skate | rank sweep (256/256, 128/128, 256/512, 512/256, 64/64, 128/512) | `--ts` rank sweep at tau 1000 | ~11:30-14:00 |
| barbel / hake / plaice | `ts16rs512tau3000_s0` / `ts24rs512tau1000_s0` / `ts16rs512tau1000_seed2` | tau 3000 / theta 24 / seed 2 | ~11:30 |
| 4090 | `big_ts16rs512tau1000_s0` | best `--ts` at 340M: -0.162 vs 340M master at 3000 | ~18:00 |
| Myriad 36472 | `x1b_dryspend_r512_lr75` | 1.3B recipe at lr 7.5e-4 (1.3B master finished 2.5868) | Sat |

## Earlier (2 Oct, 05:45)

The first `--ts` runs ended behind our rule: their leads fade late because the accumulator leaks at a fixed rate while its
steps shrink with the lr, so late in the schedule it settles below theta and flips stop. Six new runs make the leak shrink
with the lr too (`--ts_tau_anneal`).

| machine | run | what | expected |
|---|---|---|---|
| albacore / dory | `ts16rs512la300_s0` / `ts16rs512la1000_s0` | short rank 512, annealed leak tau 300 / 1000 | ~10:45 |
| elver / quillback | `ts16fullla1000_s0` / `ts16rs1024la1000_s0` | all full rank / full-rank short momentum, annealed leak tau 1000 | ~10:45 |
| flounder / inanga | `ts8rs512la1000_s0` / `ts16rs512la1000gate_s0` | theta 8 / + gate | ~10:45 |
| lamprey / goldeye | `ts16r256rs256tau1000_s0` / `ts16r128rs128tau1000_s0` | best `--ts` setting with both ranks 256 / 128 | ~11:30 |
| tope / koi | `ts16r256rs512tau1000_s0` / `ts16r512rs256tau1000_s0` | which rank matters: accumulator 256 / short momentum 256 | ~11:30 |
| barbel / hake / plaice | `ts16rs512tau3000_s0` / `ts24rs512tau1000_s0` / `ts16rs512tau1000_seed2` | tau 3000 / theta 24 / seed 2 | ~11:30 |
| 4090 | `big_ts16rs512tau1000_s0` | the best setting at 340M (4.1 s/step, 12.6 GiB). `ts16full_tau1000_s0` finished **2.6982** | ~18:00 |
| rudd / skate | `ts16r64rs64tau1000_s0` / `ts16r128rs512tau1000_s0` | both ranks 64 / accumulator 128 + short 512. Finished: `ts16rs512tau1000_s0` **2.7328**, `ts16rs512sp2_s0` 2.8138 | ~12:30 |
| harlequin / pintail / uaru | `ts16fulltau1000an05_s0` / `ts16rs512tau1000an075_s0` / `ts16r256rs256tau1000an05_s0` | milder annealing on full rank / ^0.75 at rank 512 / ^0.5 at ranks 256. Finished: `ts16rs512tau1000an05_s0` **2.7067**, `ts16rs1024tau1000_s0` 2.7203, `tsnoflip_s0` 3.1725; `...an0` stopped (+0.108) | ~13:30 |
| 4090 | `ts16full_tau1000_s0` | all full rank, tau 1000 | ~07:30 |
| lamprey / goldeye / barbel / plaice | `ts16rs512_s0` / `ts16r1024_s0` / `ts8_s0` / `ts16sp2_s0` | first-wave `--ts` | ~06:00-08:00 |
| tope / uaru | `mx_leak1000` / `tsnoflip_s0` | master leak 1000 / no-flip control | ~06:15 / ~08:00 |
| Myriad | `x1b_master_lr75` (step ~7000), `x1b_dryspend_r512_lr75` (step ~1450) | 1.3B pair at lr 7.5e-4 | ~09:00 / Sat |

## Earlier (2 Oct, 00:50): night plan

Master branch tests say master cannot do without (a) a short momentum and (b) immediate firing (RUNS.md 1 Oct 23:45).
Tonight: confirm from scratch, and test our rule rebuilt the same way (`--ts`: short rank-128 momentum for the direction,
long rank-512 accumulator of Adam-normalized steps, a trit moves when the accumulator crosses theta, spends theta).

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `ts16full_tau1000_s0` | `--ts`, all full rank (accumulator and short momentum), tau 1000. `mx_lag02` finished 3.5214 (+0.77) | ~07:00 |
| albacore / barbel / dory | `ts16_s0` / `ts8_s0` / `ts32_s0` | `--ts`, theta 16 / 8 / 32 | morning |
| elver | `ts16gate_s0` | theta 16 + sign gate | morning |
| flounder | `ts16tau1000_s0` | theta 16, accumulator leak 1000 steps | morning |
| goldeye | `ts16r1024_s0` | theta 16, full-rank accumulator | morning |
| lamprey | `ts16rs512_s0` | theta 16, short momentum rank 512 | morning |
| inanga | `fast512_dryspend_s0` | our recipe, saturated weights fire 10x faster (`--rate_peak 0.2 --g_ref 30`, same unsaturated chance) | morning |
| plaice | `ts16sp2_s0` | `--ts` theta 16, a move spends 2 theta (stays at the crossed boundary, like master without snap). Replaced `undog512_dryspend_s0`, which diverged (+1.26 at 1500) | ~06:30 |
| rudd / skate | `ts16rs512tau1000_s0` / `ts16rs512sp2_s0` | `--ts` short momentum rank 512 + leak 1000 / + spend 2 | ~06:30 |
| uaru | `tsnoflip_s0` | control: the `--ts` setup with theta 1e9 (no flip ever): what the float extras (row/column scales, magnitude adapter, qk temperature) learn alone; the `--ts` runs flip little (0.005-0.02% / step) | ~07:30 |
| tope | `mx_leak1000` | master, offset forgets with tau 1000 (tau 300 cost +0.097) | ~05:30 |
| hake | `ts16rs512tau1000an0_s0` | `--ts` short rank 512 + tau 1000, no annealing. `mx_gate` finished 2.7325 (-0.019 vs master) | ~09:30 |
| koi / quillback | `mx_b1997b2` / `mx_b2999` | master with beta1 0.997 + beta2 0.999 / beta2 0.999 alone (control). `mx_b1997` (beta1 0.997 > beta2 0.95) diverged at step ~450: Adam with beta1 > beta2 is unstable, a confound | ~06:00 |
| harlequin / pintail | `ts16rs1024tau1000_s0` / `ts16rs512tau1000an05_s0` | `--ts` full-rank short momentum + tau 1000 / short rank 512 + tau 1000 + milder annealing (the `--ts` leads fade as flips get rare late). Finished before: `undo512` 2.8360, `drywarm512` 2.8199 | ~08:30 |
| Myriad | `mx_leak300` (running), `mx_snap`, `mx_factv`, `mx_rank512`, `ours_vfull_r1024`, `x1b_dryspend_r512_lr75` (queued); `x1b_master_lr75` running | | |

## Earlier (1 Oct, 21:30): what master has that we lack

Master with one ingredient of its update taken away at a time (`bitnet/master_opt.py`, flags `--m_*`): if one
restriction alone costs master about the gap (0.05-0.07 at 110M), that ingredient is what our rule lacks.

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `mbr_*` (`master_branches.sh`, then `after_branches.sh`) | master's step-3000 state continued 500 steps with: nothing (control), leak tau 300 / 100 / 30 / 1000, snap, factored v, rank 512 / 128 first moment, clamp 1.0 / 0.6, frozen gamma, beta1 0.997, gate. Our rule lost master's lead (0.085) within 250 steps | ~23:45 |
| 4090 | `mx_lag02`, then `mx_b1997` (`mx_scratch.sh`) | master from scratch with rate-limited firing (prob. 0.02 / step) / beta1 0.997: the two branch tests that cost master +0.29-0.33 | ~02:30 / ~05:00 (after the undo bench) |
| 4090 | `undo_anatomy.py` on `big_dryspend_r512_s0` @3000 | how many flips undo takes back; how many more a stronger undo would (1 step, and 10 steps per rule) | after the branches |
| Myriad 36272 (running), 36273-36276 (queued) | `mx_leak300`, `mx_snap`, `mx_factv`, `mx_rank512` (master from scratch, one restriction each); `ours_vfull_r1024` (our rule, full rank, per-weight v) | 110M, 300M tokens, `run_110m.sh` | ~4-5 h after start |
| Myriad 36472 (queued) | `x1b_dryspend_r512_lr75` | 1.3B recipe, rank 512, float tail at peak lr 7.5e-4 (matched to master). `x1b_dryspend_r512_s0` (lr 1.5e-3) stopped 22:50 at step ~2300: +0.65 behind master at step 1750 | ~22 h after start |
| Myriad 36154 | `x1b_master_lr75` | 1.3B master at peak lr 7.5e-4 (the 1.5e-3 run diverged) | Fri 2 Oct ~08:30 |

Lab PCs: down (Thursday-evening reboot, which wipes /tmp): `drywarm512_dryspend_s0` (last seen step 3662, -0.017 vs the
rule at 3500) and `undo512_dryspend_s0` (step 3084, +0.010 at 3000) are lost and need a restart; lab121 (31 x 4070 Ti
Super 16 GB) to be used for the remaining master restrictions once the lab is back.

## Earlier (1 Oct, 19:45)

| machine | run | what | expected |
|---|---|---|---|
| pintail | `drywarm512_dryspend_s0` | memory that starts short and lengthens: -0.028 vs the rule at step 3000 | ~22:20 |
| harlequin | `undo512_dryspend_s0` | undo with the long-memory recipe: +0.017 vs the rule at step 2250 | ~22:40 |
| 4090 | `cheapcos512_dryspend_s0` | rule blended to cheap on a cosine: +0.037 vs the rule at step 8250 | ~19:55 |
| Myriad 35762 (V node, 1 A100) | `x1b_dryspend_r512_s0` | 1.3B (d2048_l24) recipe, rank 512: step 1010, 8.6 s/step | Fri 2 Oct ~15:00 |
| Myriad 36154 (queued) | `x1b_master_lr75` | 1.3B master weights at peak lr 7.5e-4 (`run_1b_master_lr75.sh`); `x1b_master` at 1.5e-3 diverged from ~33M tokens (val 4.24 -> 4.53) and was stopped at step ~1600 | ~12 h after start |

Finished today: the six flip-selection runs (flat / inv / cheap at rank 512, cheap at 256 / 128 / 64): all worse than
the rule; see RUNS.md "1 Oct 19:40".

## Earlier (1 Oct, 10:50)

| machine | run | what | expected |
|---|---|---|---|
| bufflehead | `own3000_q_dryspend_r512` | our rank-512 run from its step 3000 at 1/4 of the flip rate: -0.052 vs its full-rate self at 5750 | ~12:25 |
| mallard | `branch_m3000_q_dryspend_r512` | master @3000 on our rule at 1/4 rate | ~11:30 |
| 4090 | `big_dryspend_r1024_s0` | 340M, full rank: +0.036 vs master at step 8000 | ~11:50 |

Finished overnight: see RUNS.md "1 Oct 10:50: morning summary". Free: harlequin, shoveler, mandarin, pintail, gadwall
(Ray job), cackling (other user's server).

## Earlier (1 Oct, 02:45)

| machine | run | status | expected |
|---|---|---|---|
| 4090 | `big_master` then `big_dryspend_r1024_s0` (340M recipe at full rank, `queue_scale2.sh`) | master at step 9070 | ~02:40, then ~11:00 |
| shoveler / mandarin | `rare256sum` / `rare256flip` | step 4500: +0.116 / +0.073 behind plain rank 256 (2.8699): the rare momentum hurts | ~04:35 |
| harlequin / mallard | `tier4x64flip` / `tier4x64sum` | step ~2000: +0.052 / +0.080 behind plain rank 64 | ~05:45 |
| pintail | `gvsharp_dryspend_r512_seed2` | step 2500, -0.008 vs seed 1 | ~05:10 |
| gadwall | `rk256_int8_dryspend_s0` | int8 momentum at rank 256 (vs fp32 2.8699); shares the GPU with another user's Ray job | ~07:00 |
| bufflehead | analysis (rare_pairs, pos_gap) of the 4 runs finished overnight; shares with the Ray job | - |

cackling: another user's llama-server. bufflehead, gadwall: another user's Ray worker (1.9 GB).

## Earlier (1 Oct, 01:20)

| machine | run | what | expected |
|---|---|---|---|
| shoveler / mandarin | `rare256sum` / `rare256flip_dryspend_s0` | rank 256 + a rare-pattern rank-256 momentum (summed / own flips) vs rank 512 (2.8215) | ~04:35 |
| pintail | `gvsharp_dryspend_r512_seed2` | the best sublinear run (rank 512), seed 2 | ~05:10 |
| 4090 | `big_master` | master weights at 340M | ~02:35 |
| harlequin | `tier4x64flip_dryspend_s0` | rank 64 (dry 1/33) + three rank-64 tiers with longer memories (dry 0.016 / 0.009 / 0.005), each fed what the earlier ones miss; each tier flips on its own at 0.33x the rate (2x the flips in total). Same memory as rank 256 (2.8699) | ~05:45 |
| mallard | `tier4x64sum_dryspend_s0` | the same, summed into one flip signal | ~05:45 |

Finished: best run seed 2 2.7864 (seed 1 2.7998); dry 0.04 + spend + rank 1024 2.8156 (1/33 stays best); int8 rank 128
2.9541 (fp32 2.9501); rank 64 + refresh 3.0434 (no gain). Prepared, not launched: `--tiers` (2 x / 4 x rank 64), only if
a two-momentum run beats rank 512. Hourly checks until 10:00; no laptop GPU.

## Earlier (1 Oct, 00:40)

| machine | run | what | expected |
|---|---|---|---|
| shoveler | `rare256sum_dryspend_s0` | rank 256 + a rare-pattern rank-256 momentum (gradient outside the main subspace, dry 0.01 = longer memory), summed into the flip signal. Same memory as rank 512 (2.8215) | ~04:45 |
| mandarin | `rare256flip_dryspend_s0` | the same, the rare momentum flips on its own at 0.5x the rate (~1.5x the flips) | ~04:45 |
| bufflehead | `gvsharp_dryspend_r1024_seed2` | the best run, seed 2 | ~00:55 |
| gadwall | `gvsharp_dry04spend_r1024_s0` | dry 0.04 + spend + rank 1024 | ~00:45 |
| 4090 | `big_master` | master weights at 340M | ~02:35 |

Finished: `rk64_refresh_dryspend_s0` 3.0434 (plain rank 64 3.0401: no gain). Free: harlequin, mallard, pintail (after int8).

## Earlier (30 Sep, 20:30)

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `big_master` | master weights at 340M (reference for `big_dryspend_r512_s0` 2.7246) | ~02:30 |
| bufflehead | `gvsharp_dryspend_r1024_seed2` | the best run (2.7998), seed 2 | ~00:30 |
| mallard | `rk64_refresh_dryspend_s0` | rank 64 + subspace refresh: can low rank be rescued? | ~22:30 |
| pintail | `rk128_int8_dryspend_s0` | rank 128 with int8 momentum (vs fp32 2.9501) | ~00:20 |
| gadwall | `gvsharp_dry04spend_r1024_s0` | dry 0.04 + spend + rank 1024 | ~23:10 |
| shoveler / mandarin | `big_rk64` / `big_rk128_dryspend_s0` | 340M, rank 64 / 128: finished 2.9121 / 2.8677 | done |

Free: harlequin. cackling: another user's llama-server holds 23.5 GB (the int8 run moved to pintail).

## Earlier (30 Sep, 14:40)

| machine | run | what | expected |
|---|---|---|---|
| bufflehead | `gvsharp_dry_r1024_seed2` | the best run (dry friction + rank 1024), second seed (`--seed 2`: new init and data order) | ~18:40 |
| harlequin | `gvsharp_dryspend_r1024_s0` | dry friction + spend + rank 1024 (the two best together) | ~18:40 |
| mallard / cackling / gadwall | `rk32` / `rk64` / `rk128_dryspend_s0` | 110M rank sweep | ~15:40 |
| shoveler / mandarin | `big_rk64` / `big_rk128_dryspend_s0` | 340M, rank 64 / 128: finished 2.9121 / 2.8677 | done |
| 4090 | `big_dryspend_r512_s0` then `big_master` | 340M rank 512, then master | ~20:00, then ~12-15 h |

Finished: `gvsharp_b0995spend_r512_s0` 2.8398 (spend does not help the decay form). Free: pintail.

## Earlier (30 Sep, 13:00)

Best so far: `gvsharp_dry_r1024_s0` 2.8198, `gvsharp_dryspend_r512_s0` 2.8215 (master 2.751). Finished since 11:40:
dry + rank 1024 2.8198, dry 0.02 2.8970, beta 0.995 + rank 512 2.8339, dry + spend + rank 512 2.8215. Free: harlequin,
pintail. Down: eider (GPU), gressingham (too hot).

### Started 30 Sep, 12:05: rank sweep and model width (can the recipe fit 27B in 12-16 GB?)

Recipe = sharp base + beta 1 + dry 0.0303 + spend 3 (current best at rank 512: 2.8215). At 27B, rank 64 bf16 momentum
is ~0.95 GB (12 GB card), rank ~256 bf16 ~3.8 GB (16 GB card).

| machine | run | what | expected |
|---|---|---|---|
| mallard | `rk32_dryspend_s0` | 110M, rank 32 | ~16:00 |
| cackling | `rk64_dryspend_s0` | 110M, rank 64 | ~16:00 |
| gadwall | `rk128_dryspend_s0` | 110M, rank 128 | ~16:00 |
| shoveler | `big_rk64_dryspend_s0` | 340M (d1024_l24), rank 64 | ~01:00 |
| mandarin | `big_rk128_dryspend_s0` | 340M, rank 128 | ~01:00 |
| 4090 | `big_dryspend_r512_s0` then `big_master` | 340M, rank 512, then master | ~19:45, then ~12-15 h |

110M reference points: rank 256 2.8699, rank 512 2.8215.

## Earlier (30 Sep, 11:40)

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `big_dryspend_r512_s0` then `big_master` | scale check on the 340M model (d1024_l24): the night's winning recipe, then master, 300M tokens each (`scripts/remote/queue_scale.sh`); 3.2 s/step, 11.7 GiB | ~19:45, master after (~12-15 h) |
| lab PCs | third wave (see the night section) | dry 0.05 finished 2.8875 | ~11:45-12:20; bufflehead ~14:05 |

## Night 29-30 Sep (started 23:57-00:08, all ~04:05-04:20)

All from scratch, 300M, no look-ahead, base = Adam + gate + sharp (`--rc_scale --lr_gate --lr_vnorm 0.99 --lowrank_mag
add:16 --mag_wd 0.1 --qk_temp`). Question: why pairs seen < 1e3 times learn at half master's rate (memory horizon?).

| machine | run | change | tests |
|---|---|---|---|
| mallard | `gvsharp_b099_s0` | `--lr_beta 0.99` | memory ~100 steps instead of ~33 |
| cackling | `gvsharp_b1_s0` | `--lr_beta 1` | no friction at all (relative flip rule, no cap) |
| harlequin | `gvsharp_r512_s0` | `--lowrank 512` | control: rank is not the limit (predicted no change) |
| gadwall | `gvsharp_slowgate_s0` | `--slow_gate 64 --slow_beta 0.999` | flip only where a slow momentum agrees |
| shoveler | `gvsharp_dither_s0` | `--dither_ld` | low-discrepancy flip draw (fixed hash + step x golden ratio) |
| pintail | `gvsharp_dry_s0` | `--lr_beta 1 --dry_vec 0.0303` | dry friction on the whole vector, 1/33 of the gradient norm per step |
| mandarin | `gvsharp_b1spend_s0` | `--lr_beta 1 --spend 3` | no friction; a flip consumes the push that caused it (master's threshold crossing) |
| bufflehead | `gvsharp_rc_s0` | the base | ~01:45 |

01:30 check (val minus the base `gvsharp_rc_s0` at step 1000 / 3000 / latest): beta 1 + spend +0.40 / -0.064 / -0.073
(3250); beta 1 + dry +0.46 / -0.046 / -0.054; slow gate +0.16 / -0.051; beta 0.99 +0.13 / -0.041 / -0.043 (3500); rank 512
-0.013 / -0.028 / -0.022; beta 1 +0.65 / +0.087 / +0.054 (closing); dither +0.01 / -0.001 / +0.001 (nothing). The
long-memory runs start slower and overtake from ~66M. Base at 8500: 3.011 (master +0.241).

Queued on bufflehead after the base (~01:45 -> ~05:40): `gvsharp_b1spend_slowgate_s0` = beta 1 + spend 3 + slow gate
(r64, 0.999): the two best new mechanisms together.

03:30 check (vs base at the latest step): beta 1 + dry -0.102 (7750); beta 0.99 -0.078 (8500); beta 1 + spend -0.061
(8000, was -0.088); beta 1 + spend + slow gate -0.055 (4000); slow gate -0.034 (7750); rank 512 -0.024; dither +0.002;
beta 1 +0.043. Memory length is the lever.

Next wave, queued in each runner (starts when the current run ends, ~03:45-04:20, ends ~08:00-08:30):

| machine | run | change |
|---|---|---|
| mallard | `gvsharp_b0995_s0` | `--lr_beta 0.995` (longer than 0.99) |
| cackling | `gvsharp_dry01_s0` | beta 1 + `--dry_vec 0.01` (weaker friction: longer memory) |
| gadwall | `gvsharp_dry1_s0` | beta 1 + `--dry_vec 0.1` (stronger friction: shorter memory) |
| shoveler | `gvsharp_dryspend_s0` | beta 1 + dry 0.0303 + spend 3 |
| mandarin | `gvsharp_b099spend_s0` | beta 0.99 + spend 3 |
| harlequin | `gvsharp_dryw_s0` | beta 1 + `--dry_w 0.0303` (dry friction per weight, new flag) |
| pintail | `gvsharp_b099slow_s0` | beta 0.99 + slow gate |

05:30 check (vs base): dry + spend -0.093 (3500; dry alone -0.046 at 3000); beta 0.995 -0.084 (4000); beta 0.99 + spend
-0.067 (3750); dry 0.01 -0.027 (4000, slow start); dry per weight -0.013 (3250); beta 0.99 + slow gate -0.024 (3250);
dry 0.1 +0.018 (3500: too short a memory); beta 1 + spend + slow gate -0.027 (8500). Queued on bufflehead after it
(~05:45 -> ~09:40): `gvsharp_dry_r512_s0` = the best (dry friction) + rank 512 (does a long memory need more rank?).

06:30 check (vs base): dry + spend -0.102 (6000; dry alone -0.085); beta 0.995 -0.095 (6500); beta 0.99 + spend -0.078;
dry per weight -0.051 (5750); dry 0.01 -0.043; beta 0.99 + slow gate -0.042; dry 0.1 +0.020. beta 1 + spend + slow
gate finished 2.9708.

07:30 check (vs base): dry + spend -0.119 (8250); beta 0.995 -0.111 (8750); **dry + rank 512 -0.110 at 3750** (dry alone
-0.054 at 3250: with a long memory, rank helps); beta 0.99 + spend -0.085; dry per weight -0.050; beta 0.99 + slow
gate -0.049; dry 0.01 -0.035; dry 0.1 +0.015.

Third wave, queued in each runner (starts ~07:35-08:30 as the second wave ends, ends ~11:30-12:30):

| machine | run | change |
|---|---|---|
| mallard | `gvsharp_dryspend_r512_s0` | beta 1 + dry 0.0303 + spend 3 + rank 512 (the three best together) |
| cackling | `gvsharp_b0995_r512_s0` | beta 0.995 + rank 512 |
| gadwall | `gvsharp_b0998_s0` | beta 0.998 (the decay sweep: 0.97 / 0.99 / 0.995 / 0.998 / 1) |
| harlequin | `gvsharp_dry_r1024_s0` | dry 0.0303 + rank 1024 (does rank keep helping?) |
| shoveler | `gvsharp_b0995spend_s0` | beta 0.995 + spend 3 |
| mandarin | `gvsharp_dry05_s0` | beta 1 + dry 0.05 (friction sweep: 0.01 / 0.02 / 0.0303 / 0.05 / 0.1) |
| pintail | `gvsharp_dry02_s0` | beta 1 + dry 0.02 |

08:30 check: second wave finished (dry + spend 2.8699 best; beta 0.995 2.8783). Third wave running (steps 750-1970):
dry + spend + rank 512 -0.123 at 1750; dry 0.05 -0.074; beta 0.995 + rank 512 -0.045; beta 0.995 + spend -0.036.
dry + rank 512 (bufflehead) -0.131 at 6000.

09:30 check: dry + rank 512 -0.151 at 8000 (2.8808, ends ~09:55); dry + spend + rank 512 -0.158 at 4000; beta 0.995 +
rank 512 -0.120; dry + rank 1024 -0.103 (3250, not ahead of rank 512); beta 0.995 + spend -0.097; dry 0.05 -0.082; beta
0.998 -0.076; dry 0.02 -0.027 (3000). Queued on bufflehead after dry + rank 512 (~09:55 -> ~13:50):
`gvsharp_b0995spend_r512_s0` = beta 0.995 + spend 3 + rank 512 (the decay form of the best combination, to pick between
decay and dry friction for goldbug).

10:50 check (last): dry + rank 512 finished 2.8356. Running (vs base): dry + spend + rank 512 -0.160 (7000); beta 0.995 +
rank 512 -0.143; dry + rank 1024 -0.140 (6000); beta 0.995 + spend -0.100; dry 0.05 -0.096; beta 0.998 -0.090; dry 0.02
-0.071 (6000); beta 0.995 + spend + rank 512 -0.111 (1750). Ends ~12:15-12:45 (bufflehead ~13:50). No more checks.

Stopped: `gvundo_rc_s0` (undo reverses 2% of flips with the gate: inert). Down: eider, 4090. gressingham too hot (91 C).
A fixed hash dither (flip iff u_ij < p) would freeze 98% of weights (p <= rate = 0.02), hence the low-discrepancy form.

## Earlier (29 Sep, 23:10)

| machine | run | what | expected |
|---|---|---|---|
| mallard | `gvundo_rc_s0` | Adam + gate + undo (restarted from scratch at 22:55; the 4090 is unbookable and eider's GPU dropped off the bus) | ~02:35 |
| bufflehead | `gvsharp_rc_s0` | Adam + gate + sharp | ~01:45 |

At step 3000 sharp is 3.409 vs Adam + gate 3.468 (master 3.186). Undo at step 500: 4.743 vs 4.670 (early).
Down: eider (GPU gone, needs a reboot), 4090 (booking pending).

## Earlier (29 Sep, 22:05)

Only two runs, both no look-ahead, from scratch, row/column scales (all other runs and tests stopped at 21:40):

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `gvundo_rc_s0` | Adam + gate + undo | ~00:30 |
| bufflehead | `gvsharp_rc_s0` | Adam + gate + sharp (additive r16 + adapter weight decay + per-head temperature); moved off harlequin, where another user's Ray job halved its speed | ~01:45 |

Free: gressingham (GPU at 90 C), mallard, harlequin (another user), cackling, shoveler / pintail (other users), laptop.
Names: sharp = `--lowrank_mag add:16 --mag_wd 0.1 --qk_temp`; undo = `--undo`.

## Now (29 Sep, 18:00)

| machine | now | then | expected |
|---|---|---|---|
| 4090 | swing tests (nola_lab @4500): fu1 beta 0.97 + cap + tanh (done: -0.032, swing kept), fu2 + undo, fu3 beta 0.9 + undo, fu4 plain + undo, fu5 all + gate + vnorm | `user_cap1_flip15_s0` (cap 1, flip scale 1.5: flips ~ plain) | arms ~18:05-19:05; training ~20:20 |
| pintail | `user_cap3_s0` (user's rule, cap 3) | - | ~19:50 |
| mallard | swing tests: plain + undo, dry friction on the vector (1/33), dry friction per weight (1/33); `user_cap1_s0` stopped (flip scale 10: 5-7x too few flips) | - | ~18:25, ~18:50, ~19:15 |
| cackling | `grav05_rc_s0` | - | ~20:40 |
| harlequin | `grav05_gatevnorm_rc_s0` | - | ~20:40 |
| gressingham | damping tests: plain (done: every step's whole move is uphill one step later), gate+vnorm @4500, gate / gate+vnorm @8000 | - | ~19:00 |
| laptop | idle; next: dry friction per weight + undo once the undo code has run | - | - |

## Now (29 Sep, 15:45): the user's rule

Momentum as one velocity vector: `--lr_beta 1` (no friction; the gradient is the gravity), `--mom_ncap C` (cap the
whole momentum at C x a slow EMA of the total gradient norm), `--pfun_tanh 10` (flip chance rate * tanh(|M| / (3 v0)),
v0 = 10 x a slow EMA of the layer's mean |g|: absolute units). All from scratch with row/column scales.

| machine | run | what | expected |
|---|---|---|---|
| 4090 | `usersweep_cap3` -> `cap10` -> `cap30` -> `cap100` -> `nocap` | short sweep of the cap, stop at step 1500 (~50M) | ~25 min each, to ~18:00 |
| pintail | `user_cap10_s0` | full 300M, cap 10 | ~19:10 |
| mallard | `user_cap30_s0` | full 300M, cap 30 (shoveler's GPU was taken by another user) | ~19:15 |

## Night 28-29 Sep (planned 01:10, runs to ~10:00)

Swing test, fixed divisor (4090, 25%): flips did not drop (164k -> 178k per step), swing unchanged (lag 3 -0.241),
held-out -0.0034 vs -0.0108 for the old rule. A per-layer divisor cannot see the swing (it lives in a few
directions); per weight the flip chance already follows |M_ij|. Next: per-direction arms.

Found: without look-ahead the trainer ignored `--lr_gate` and `--lr_vnorm` (wired only into the look-ahead path).
Fixed; `--speed_row` added (per-row speed reference). Runs are list-driven (`scripts/night/*.list`, re-read before
each run; edit the lists on the remote home, not by rsync from the laptop).

| machine | now | next (list) | expected |
|---|---|---|---|
| 4090 | `vnorm_rc_s0` done **3.1063** (0.98x flips); `gatevnorm_rc_s0_seed2` running | - | ~12:30 |
| pintail | `rate085_rc_s0` done **3.1208** (= the gate); `vnorm064_rc_s0` (Adam step at 0.64x flips, no gate) started 10:45 | - | ~14:20 |
| shoveler | `gate_rc_s0_seed2` done **3.1138**; `rate064_rc_s0` (gate + vnorm's rate control) started 09:30 | - | ~13:10 |
| laptop | probes done (rc_s0 / speedref_rc_s0 @5000, @8000); now gate_rc_s0 @5000 (gate rule) and rc_s0 @5000 with per-weight analysis | - | ~07:50, ~09:10 |
| 4090 (next to training) | swing tests: gate rate control (old rule x0.62), gravity 0.5, gradient units beta 0.8, gravity 0.8 (relaunched 03:28: first launch died silently) | - | ~03:55, ~04:20, ~04:45, ~05:10 |

First per-weight answer (12-step check, old rule): 45% of flips move uphill on the true gradient; at a reversal a
weight's momentum is ~7x its true gradient (~7 steps to cross zero); 40% of reversing weights never follow within
the window; only 13.5% of the momentum's size lies in the span of the true gradients (the rest is batch noise).
Asymmetric gravity (user's idea): where the batch gradient opposes a weight's momentum, decay it with 0.5 instead of
0.97. **If its swing test damps the swing: add `--gravity` to lowrank_step (per-weight on the rebuilt M before the
rank-r compression) and put `gravity_rc_s0` first on the 4090 list.**


Swing test, old rule (4090, 25%): held-out 3.5495 -> 3.5387 (-0.0108) over 41 steps, 165k flips every step, swing as
before (lag 3 -0.252, momentum vs next gradient -0.172). Lab PCs rebooted (Mon/Thu) 20:00-00:07; `nola_lab`
ckpt_5000 is gone with /tmp, so the swing tests use ckpt_4500. The laptop's beta 0.5 test was stopped (beta 0.8
made the swing faster: lag 2 -0.294, momentum -0.207 now).

All other lab PCs released. Finished 16:40-17:30 (val at 300M; plain `nola_lab` 3.158, master 2.759):
`evid3_b131` 3.0970, `rc_b131` 3.1480, `adaptrate_b131` 3.1604, `multibeta_b131` 3.1926, `small_step_s0` 3.1933,
`master_q4_b131` 2.9231, `master_q3_b131` 3.1222, `master_q2_b131` 3.4686. Stopped: `select_b131` (3.366 at 229M).

Stopped at 15:10 (step-size runs other than 1/4 from scratch): `small_step8_lab`, `small_step8_s0`, `small_step8_b131`,
the queued `mech_user_q_b131`. Finished: `small_step_b131` 3.1551, `accum33_b131` 3.2795.

Home-quota incident 15:20: the syncs shipped every checkpoint through the home folder (10 GB) and filled it; all
syncs now ship logs only, checkpoints stay on each machine. VirtualBox.xml (truncated by the full disk) restored
from VirtualBox.xml-prev.

## Finished overnight (27-28 Sep)

300M runs (val loss at 300M; references: master 2.759, look-ahead + additive r4 3.082, look-ahead baseline 3.127,
no look-ahead 3.158):

| run | what | val @300M |
|---|---|---|
| `la_sched` | look-ahead 0-30M, off to 131M, on after | **3.1108** |
| `la_sched98` | look-ahead 0-30M, off to 98M, on after | 3.1135 |
| `mech_user_b131` | your design (gain 1), no look-ahead, branch of `nola_lab` from 131M | **3.1202** |
| `mech_v1_b131` | V1 target point, no look-ahead, branch from 131M | 3.1215 |
| `nola_then_la` | no look-ahead to 131M, then look-ahead | 3.1225 |
| `mech_user_g0_b131` | your design without the move correction, branch from 131M | 3.1325 |
| `magadd16_qk_lab` | look-ahead + additive r16 + temperature, to its planned 200M | 3.1212 @200M |

From scratch to 131M (no look-ahead; references at 131M: `nola_lab` 3.573, look-ahead baseline 3.345):

| run | val @131M |
|---|---|
| `mech_user_s0` (your design, gain 1) | **3.505** |
| `mech_user_g0_s0` (your design, gain 0) | 3.548 |
| `mech_v1_s0` (V1) | 3.582 |

Bench, corrected move correction (no-look-ahead checkpoint, held-out loss change after 66 steps; plain momentum
-0.024, V1 -0.047): V2 gain 1 **-0.068**, gain 3 +0.098; V3 (bench version, unfloored curvature) gain 1 +0.89,
gain 3 +0.53 (diverges).

Toy (induction nats at 15k steps, seed 1 / seed 2): plain momentum 1.07 / -0.02; V1 6.49 / 0.07; your design
2.68 / 1.52; your design gain 0 0.44 / 2.44.

## The mechanisms (`--mech`)

- **V1 (target point, as tested on the bench):** D = displacement to the minimum; each step D <- 0.97 (D - move)
  + 0.03 (-g / h), h measured once. On the bench (no-look-ahead checkpoint, warm momentum): held-out loss -0.047
  after 66 steps vs -0.024 for plain momentum (2x), same number of flips. Logging shows the carried part
  dominates the new evidence (~10^2-10^6 x): in practice it heads for its first target estimate and stops where it
  has covered the distance.
- **user (your design):** target point from the most recent gradients (short memory 0.8, target memory 0.9,
  reversible); curvature re-measured every step (same-batch secant, running averages, floored); the remembered
  direction F corrected by the measured effect of each move (gain 1); when F disagrees with the target
  (cos < 0), F is rotated onto it. Flips from F. `gain 0` = without the move correction.

## Fixes made tonight

- Bench: the first rotation test used a look-ahead checkpoint with empty momentum and mostly measured the
  look-ahead switch-off shock; redone on the no-look-ahead checkpoint with its own momentum.
- Move correction: gain 33 (every remembered gradient shifted by the one measured change) diverges (+0.85 held-out
  loss in 33 steps); the trainer uses gain 1 (and gain 0 as the hedge).
- User design: one step's curvature can be ~0 and blow the target up (10^8); curvature from running averages,
  floored, target clamped to +-2 trit steps.
- 4090 test output was buffered by `grep` (invisible until the end); lab scripts use `--line-buffered`.
- Plots assumed 32,768 tokens per step for every run (batch 48 was drawn at a third of its text); fixed.
- `pgrep` patterns that matched their own command line (two stalls, one self-kill); wait patterns now match only
  the training processes' `--out_dir ...$`.

## Started, stopped, not queued

| run | what | stopped at / planned | finish it? |
|---|---|---|---|
| `nola_add16` | momentum without look-ahead + additive r16 | 112M / 300M | no: the adapter takes over the q/k projections, loss stuck (~4.6) |
| `qkprot60` | per-head temperature + head protection | 28M / 60M | no: approach dropped, checkpoint gone |
| `overnight_full` | look-ahead only, no momentum | 131M / 300M | no: superseded |
| `armA_cos_la1` | stateless look-ahead | 202M / 300M | no: superseded |
| `s20_maskstuck` | stuck-mask screen | 15M / 20M | no: stopped because it was worse |
| `curve_master` | master with window-curve measurements | 208M / 300M | no: `master_tracked` covers 300M |
| `la40_off30` | look-ahead first 40 steps, then off | 30M | only for its full curve (it has joined the no-look-ahead curve) |

## Never run

Scripted, never started:

| run | what | why not |
|---|---|---|
| `all_fixes_full` | every fix together, 300M | removed from the queue |
| `mul60_heads` | multiplicative magnitude to 60M, heads measured | removed |
| `gate_full` | sign gate, 300M | replaced by additive magnitude |
| `s60_gate`, `s20_gate_maskstuck` | gate screens | dropped when the laptop queue changed |

Proposed, not built:

| idea | status |
|---|---|
| low-rank versions of the mechanisms | after the night's results |
| output bias so the unigram frequencies don't have to be built by the trits | proposed (the no-look-ahead plateau) |
| copy-data curriculum | not yet (user) |
| toy: adapter + weight decay without look-ahead; adapter at flip rate 0.005 | proposed |
| 600M run; MLP width sweep (N+K scaling); multi-GPU trainer; FineWeb data; flip-rate schedule shaped like the LR schedule | later |
| softer look-ahead filter | moot: no look-ahead from now on |
