# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16). From
2026-09-27 on, new runs are without look-ahead.

Last updated: 2026-09-29 22:05

Run names: `rc` = learned row/column scales, `s0` = from scratch (step 0), `b131` = branch from `nola_lab` at 131M,
`gate` = sign gate (flip only where this batch's gradient agrees with the momentum), `vnorm` = factored Adam step
(momentum divided by a row x column gradient size), `rateNNN` = plain momentum at 0.NNx peak flip rate,
`speedref` = per-layer slow speed reference, `seed2` = same run with seed 2.

## Now (30 Sep, 14:40)

| machine | run | what | expected |
|---|---|---|---|
| bufflehead | `gvsharp_dry_r1024_seed2` | the best run (dry friction + rank 1024), second seed (`--seed 2`: new init and data order) | ~18:40 |
| harlequin | `gvsharp_dryspend_r1024_s0` | dry friction + spend + rank 1024 (the two best together) | ~18:40 |
| mallard / cackling / gadwall | `rk32` / `rk64` / `rk128_dryspend_s0` | 110M rank sweep | ~15:40 |
| shoveler / mandarin | `big_rk64` / `big_rk128_dryspend_s0` | 340M, rank 64 / 128 | ~22:00 |
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
