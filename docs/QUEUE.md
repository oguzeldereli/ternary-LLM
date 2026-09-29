# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16). From
2026-09-27 on, new runs are without look-ahead.

Last updated: 2026-09-29 15:45

Run names: `rc` = learned row/column scales, `s0` = from scratch (step 0), `b131` = branch from `nola_lab` at 131M,
`gate` = sign gate (flip only where this batch's gradient agrees with the momentum), `vnorm` = factored Adam step
(momentum divided by a row x column gradient size), `rateNNN` = plain momentum at 0.NNx peak flip rate,
`speedref` = per-layer slow speed reference, `seed2` = same run with seed 2.

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
