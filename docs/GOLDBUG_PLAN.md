# Goldbug plan (booking 8-10 Oct, 3 days, 2 x RTX PRO 6000 96 GB)

Written 2 Oct. Status of the method: two-timescale threshold flips (TTF, `--ts`) beat master weights at 110M even when
master gets our float extras (ranks 512: 2.702 vs 2.730, two seeds each), beat 340M master (2.6268 vs 2.6626) and run on
the 1.3B master's curve so far (1.3B at step 1750: 3.291 vs master 3.301). See RUNS.md, 2 Oct.

## Goal

One decisive result at a scale where the claim matters: **TTF vs master weights at 1.3B on a larger, cleaner corpus
(FineWeb-Edu), several times more tokens than so far**, both with the same float extras, same data order, same schedule.
Both GPUs run the pair side by side (GPU 0: TTF, GPU 1: master + extras); no exploration on the booking.

## Budget

- 1.3B TTF on one A100 now: 9.7 s / step at 32k tokens = ~3.4k tokens/s (old rule 9.2 s, master 4.7 s; the 1.3B A100
  profile, Myriad 40272-40274, says where the time goes). An RTX PRO 6000 has far more fp32 throughput than an A100
  (our momentum update is fp32) and ~1.5-2x the bf16; assume **5-8k tokens/s per GPU** until measured.
- ~66 usable hours (setup and copy-off take the rest): **~1.2-1.9B tokens per run** at 1.3B (4-6x the 300M so far).
  Fix the token budget at the start from a 20-minute throughput test, so both runs finish their cosine schedule.
- Memory is not a constraint at 96 GB (1.3B TTF peaks at 24.7 GiB at 32k tokens/step): use 64k tokens per step
  (batch 32 x 2048) if the throughput test shows it is not slower per token; lr then as tuned at 32k (7.5e-4) unless a
  short sweep says otherwise.

## Before the booking (3-7 Oct)

| when | what | why / done when |
|---|---|---|
| Sat 3 | storage-format ablation results (17 runs, ~00:15); 1.3B TTF final (~14:45); 1.3B A100 profile | pick the state format; confirm the 1.3B result; know the slow kernels |
| Sat 3 | **save the full TTF state in checkpoints** (short momentum m and v are not saved now: a resume re-initialises them) and test resume mid-run | a 66-hour run must survive a crash |
| Sat 3 | **gradient accumulation for TTF** (capture mode keeps only the last micro-batch's gradient now) or confirm 64k tokens fit without it | bigger batches if wanted |
| Sat-Sun | **FineWeb-Edu data**: `prep_fineweb.py` (from `prep_wiki.py`, Llama-2 tokenizer), tokenize ~2B tokens (sample-10BT) on a lab PC's /tmp (~4 GB uint16), plus a held-out validation shard; keep the wiki val for continuity | data ready, bits per byte comparable |
| Sun 4 | 340M TTF ranks 256 / 128 (two runs) | does the rank penalty shrink with width (the 27B question) |
| Sun 4 | **Blackwell environment**: setup script with a CUDA 12.8+ torch build (the cu126 wheels we use do not support sm_120) and a Triton smoke test of our kernels; fallback flags (`--dw_mode` / no `--int8`) if a kernel fails | the booking cannot be lost to setup |
| Mon 5 | speed fixes from the profile (e.g. batched QR over layers, TF32, kernel configs) | more tokens per hour |
| Mon 5 (lab reboot in the evening) | keep lab runs short that day | |
| Tue 6 | **dry run of the exact goldbug queue** on the 4090 at 340M for ~1 h: data loading, both arms, checkpoint every 2 h, copy-off, resume, throughput log | everything unattended works |
| Wed 7 | freeze and tag the code (`goldbug-v1`); end the 4090 booking and copy its /scratch0 runs to the laptop (one workstation booking at a time) | |

## On the booking

| hour | GPU 0 | GPU 1 |
|---|---|---|
| 0-1 | env setup (script), copy data, 20-min throughput test of both arms, set the token budget | same |
| 1-~66 | **TTF 1.3B** (ranks 512, tau 1000, steps x ratio^0.5, theta 16, fused, + extras), lr 7.5e-4 | **master 1.3B + the same extras**, lr 7.5e-4 |
| every 2 h | checkpoint; copy logs + metrics off the machine (laptop pulls via knuckles) | same |
| ~66-72 | final eval (wiki val and FineWeb val, loss and bits per byte), copy final checkpoints to the laptop | same |

Unattended: one queue script per GPU (`CUDA_VISIBLE_DEVICES`), resume on restart, a watchdog that restarts a crashed
run from its last checkpoint; hourly checks from here.

## Decision points

- If the 1.3B TTF run (Myriad, Sat) falls behind 1.3B master late, as the 110M runs do in the cosine tail: use the
  milder annealing that holds the late slope (already in the 1.3B run) and decide between the pair above and two TTF
  seeds.
- If the storage ablation finds a free format (e.g. int8 direction + fp16 accumulator), use it on goldbug so the
  memory claim (~1 byte per parameter at 27B) is backed by a large run; otherwise fp32 state (memory is not the limit).
- If the Blackwell kernels fail: the non-int8 kernel path, then plain torch matmuls; measure the cost in the first hour.

## Risks

- Throughput unknown on Blackwell; Triton kernels untested there.
- Short-momentum state not saved in checkpoints (fix before).
- /scratch0 is wiped when the booking ends: copy-off is part of the queue, not a manual step.
- A single seed per arm: the seed spread at 110M is ~0.01, smaller than the expected gap; say so in the write-up.
