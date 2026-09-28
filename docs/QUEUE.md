# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16). From
2026-09-27 on, new runs are without look-ahead.

Last updated: 2026-09-28 01:45

## Night of 27-28 Sep (until ~10:30)

Order: the new momentum mechanisms first, then unfinished runs that were already running, then other unfinished
runs. `--mech` runs keep their mechanism state full-size (one fp32 value per weight) to judge the mechanisms;
a low-rank version follows if one works. Their state is not saved in checkpoints (a resume restarts it warm
from the saved low-rank momentum).

| machine | # | run | what | tokens | expected |
|---|---|---|---|---|---|
| **4090** (needs `start.sh night_4090.sh`) | 1 | `mech_user_b131` | your design, move-correction gain 1: branch of `nola_lab` at 131M -> 300M | 131 -> 300M | ~04:35 |
| | 2 | `mech_user_s0` | your design, gain 1, from scratch | 0 -> 131M | ~06:50 |
| | 3 | `la_sched` (resume) | look-ahead 0-30M, off to 131M, on after | 215 -> 300M | ~08:50 |
| | 4 | `nola_b48` (resume, filler) | no look-ahead, batch 48 | step 1550 -> 9155 | after 10:30 |
| **shoveler** | 0 | `nola_then_la` (finishing) | no look-ahead to 131M, then look-ahead | -> 300M | ~01:50 |
| | 1 | `mech_v1_b131` | V1 target point: branch of `nola_lab` at 131M -> 300M | 131 -> 300M | ~03:55 |
| | 2 | `mech_v1_s0` | V1 from scratch | 0 -> 131M | ~05:35 |
| | 3 | `magadd16_qk_lab` | look-ahead + additive r16 + temperature, to its planned 200M | 181 -> 200M | ~06:10 |
| | 4 | `magadd16_wd_qk_lab` | the same + adapter weight decay (best loss so far) | 131 -> 300M | ~11:25 |
| **goosander** | 0 | full mechanism test (finishing) | V0-V3 on the no-look-ahead bench, 66 steps | - | ~01:50 |
| | 1 | `mech_user_g0_b131` | your design without the move correction (gain 0: target point + rotation), branch at 131M | 131 -> 300M | ~05:10 |
| | 2 | `la_sched98` (resume) | look-ahead 0-30M, off, on again from 98M | 159 -> 300M | ~09:30 |
| **laptop** | 1 | `mom_mech_v1`, `mom_mech_user`, `mom_mech_user_g0` | the mechanisms on the 2-layer induction toy (no look-ahead, 15000 steps) | toy | ~04:40 |

Comparison for the branches: `nola_lab` itself (the same run with plain rank-256 momentum), val 3.573 at 131M,
3.158 at 300M. For the from-scratch runs: `nola_lab` (plateau at 2-5M) and the look-ahead baseline.

Scripts: `scripts/remote/night_4090.sh`, `scripts/lab/night_shoveler.sh`, `scripts/lab/night_goosander.sh`,
`scripts/lab/mech_run.sh`, `scripts/toy/mech.sh`; the mechanisms are `mech_step` in `bitnet/train.py`.

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
