# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16). From
2026-09-27 on, new runs are without look-ahead.

Last updated: 2026-09-28 15:00

## Now (28 Sep, 13:50)

| machine | run | what | expected |
|---|---|---|---|
| **4090** | `accum33_b131` | accumulate 33 steps without flips, then flip a loss-chosen random subset of the entries whose accumulated push stands out (|M| > 3 rms); branch 131M -> 300M | ~15:00 |
| | `accum33_s0` | the same from scratch -> 300M | ~17:30 |
| **goosander** | `small_step_b131` | plain momentum at 1/4 of the flip rate, branch 131M -> 300M | ~14:30 |
| **cackling** | `small_step_s0` | plain momentum at 1/4 of the flip rate, from scratch -> 300M | ~15:30 |
| | `mech_user_q_b131` | your design (gain 1) at 1/4 of the flip rate, branch 131M -> 300M | ~18:00 |
| **mallard** (third lab PC, added 13:40) | `small_step8_lab` | plain momentum at 1/8 of the flip rate, branch 131M -> 300M | ~16:00 |
| **mandarin** (fourth lab PC, added 14:10) | `small_step8_s0` | plain momentum at 1/8 of the flip rate, from scratch -> 300M | ~17:30 |
| **bufflehead** (lab PC) | `select_b131` | online-learned flip selector: propose at 2x the rate, an MLP keeps the best half (labels: do the next 8 batches' gradients still push the same way); branch 131M -> 300M | ~16:30 |
| **ruddy** (lab PC) | `adaptrate_b131` | flip rate adapted to keep cos(g, M) near +0.03, bounded 1/16x-2x; branch 131M -> 300M | ~16:30 |
| **eider** (lab PC) | `multibeta_b131` | three momenta (decay 0.8 / 0.95 / 0.99), per layer the best recent predictor flips; branch 131M -> 300M | ~17:00 |
| laptop | `small_step8_b131` | the same 1/8 branch as mallard, slower copy (can be stopped) | ~18:00 |

Early numbers (val): at step 4750 the 1/4-rate branch is at 3.342 vs 3.497 for the same run at the trained rate and
3.457 for your design at the trained rate; accumulate-then-flip 3.443 at step 4250 (trained rate 3.546).

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
