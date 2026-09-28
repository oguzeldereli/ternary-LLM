# Run status: running, queued, unfinished, never run

Kept current: updated whenever a run starts, finishes, stops or is queued. Finished runs move to
[RUN_INDEX.md](RUN_INDEX.md) / [RUNS.md](RUNS.md). Machines: **4090** (booked workstation beachcomber, started
by the user with `bash ~/ternary-LLM/scripts/remote/start.sh SCRIPT`), **shoveler** / **goosander** (lab 3090 Ti
PCs, at most two in use), **laptop**. Token counts are training tokens (32,768 per step at batch 16).

Last updated: 2026-09-28 01:05

## Running now

| run | what | where | at / planned | expected | script |
|---|---|---|---|---|---|
| `nola_then_la` | momentum without look-ahead to 131M (branch of `nola_lab`), then look-ahead on | shoveler | 289M / 300M | ~01:15 | `scripts/lab/queue_shoveler.sh` |
| `la_sched98` | look-ahead 0-30M, off, on again from 98M | goosander | 156M / 300M | ~05:20 | `scripts/lab/la_sched98.sh` |
| `nola_b48` | momentum without look-ahead, batch 48 (the text a look-ahead step reads) | 4090 | step 1550 / 9155 | ~08:30 | `scripts/remote/nola_b48_now.sh` |
| momentum mechanism test, fast | V0 plain / V1 target point / V2 transport / V3 both, 33 steps, on the no-look-ahead bench | shoveler | computing true gradient | ~01:15 | `scripts/lab/momentum_mech_fast.sh` |
| momentum mechanism test, full | same, 66 steps | goosander, 4090 | running | ~01:30 | `scripts/lab/momentum_mech.sh`, `scripts/remote/momentum_mech.sh` |

`nola_b48` has answered its question (at equal text read, batch 48 = batch 16 without look-ahead, both better
than look-ahead); proposed to stop.

## Queued

| run | what | where | at / planned | after | 
|---|---|---|---|---|
| `la_sched` | look-ahead 0-30M, off to 131M, on after (paused) | 4090 | 215M / 300M | `nola_b48` |
| `magadd16_qk_lab` | look-ahead + additive r16 + per-head temperature | shoveler | 181M / 200M (its plan) | `nola_then_la` |
| `magadd16_wd_qk_lab` | look-ahead + additive r16 + adapter weight decay 0.1 + temperature (best loss so far) | shoveler | 131M / 300M | `magadd16_qk_lab` |

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
| `all_fixes_full` | every fix together (momentum + look-ahead + additive r16 + gate + vnorm + spend + refresh + temperature), 300M | removed from the queue |
| `mul60_heads` | multiplicative magnitude to 60M, heads measured | removed |
| `gate_full` | sign gate, 300M | replaced by additive magnitude |
| `s60_gate`, `s20_gate_maskstuck` | gate screens | dropped when the laptop queue changed |

Proposed, not built:

| idea | status |
|---|---|
| full momentum design: target point from the most recent gradients (can reverse) + correction of the remembered direction by the measured effect of each move + detection of geometry change | next, after the preliminary mechanism test |
| output bias (a float vector over the vocabulary) so the unigram frequencies don't have to be built by the trits | proposed (the no-look-ahead plateau) |
| softer look-ahead filter | moot: no look-ahead from now on |
| copy-data curriculum | not yet (user) |
| toy: adapter + weight decay without look-ahead; adapter at flip rate 0.005 | proposed |
| 600M run; MLP width sweep (N+K scaling); multi-GPU trainer; FineWeb data; flip-rate schedule shaped like the LR schedule | later |
