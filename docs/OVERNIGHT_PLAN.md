# Overnight plan, 3 Oct (local only)

User, 3 Oct ~00:20: "Start evaluation process, start running our predecessors, start optimizing memory usage so that we
can get the headline of fitting 27B etc, start checking all code everywhere for bugs and so on, make sure to evaluate
our results so that no one can discredit or attack them in any way, distribute this work overnight."

Work is done in half-hour slots (:17 and :47, 00:47-08:47); each slot takes the first item that is not done, works on
it for one bounded chunk, writes what it found under the item, and commits locally (never push). The 09:17 slot writes
the morning summary. GPU work only on lab PCs / 4090 / Myriad (never the laptop GPU; CPU tests on the laptop are fine).

## Runs started 00:20-00:30 (robustness; compare against the numbers in brackets)

All waiters first hung on a self-matching `pgrep -f bitnet.train`; restarted 00:27 with `"[p]ython -m bitnet.train"`. ETAs
below are from 00:30.

| machine | run | why (attack it answers) | compare with | ETA |
|---|---|---|---|---|
| barbel / flounder / goldeye | `mx_lr075_s0` / `mx_lr3e3_s0` / `mx_lr6e3_s0` | "master baseline under-tuned": master lr 7.5e-4 / 3e-3 / 6e-3 (only 1.5e-3 so far) | master 2.7513 | ~05:00 |
| lamprey / quillback | `mx_seed2` / `mx_seed3` | "one master seed": error bars for the reference | master 2.7513 | ~05:00 |
| hake | `ts16rs512tau1000an05_seed3` | third seed of the headline TTF run | 2.7067 / 2.6978 | ~05:00 |
| koi | `tsbare_rs512an05_s0` | "the float extras do the work": TTF without row/col scales, adapter, qk temperature | master (bare) 2.7513 | ~05:00 |
| tope / rudd | `fw_ts512an05_600M` / `mx_fw_600M` | "advantage vanishes with more tokens / one dataset": FineWeb-Edu, 600M tokens, no repetition | each other | ~10:00 |
| dory / uaru | `big_ts16r256rs256tau1000_s0` / `big_ts16r128rs128tau1000_s0` | rank at width (27B) | 340M r512 2.6268, master 2.6626 | Sat evening |
| albacore / elver | `qm8a16_seed2` / `qmf8a16_seed2` | storage "free" beyond noise | fp32 seed 2 2.6978 | ~04:50 |
| 4090 / harlequin / pintail | `qfp16s_s0` / `qm8a16v8_s0` / `qm8a8det_s0` | storage formats | see QUEUE.md | ~04:30 |
| inanga | `orth_debug.sh` | Cholesky-QR | Householder | ~01:00 |

Note: the training-time eval windows are seeded by the run's seed (`tc.seed + 12345` in `evaluate`), so seed-2/3
runs are scored on different val windows from seed-0 runs. Paper numbers must come from the final evaluation (E1).

## Work items (in order)

Status: todo / in progress / done (with a short result).

1. **C1 Checkpoint inventory and rescue** (todo). Lab /tmp is wiped at the Monday reboot. List final checkpoints of every
   headline run (110M master, master + extras x2, TTF r512 x2-3, full rank, r256, int8, storage finals, 340M TTF and
   master, 1.3B runs on Myriad, the old rule) with host and path; copy the headline ones to the laptop
   (`checkpoints/final/<run>/`), directly machine-to-laptop (never through the UCL home).
2. **E1 Final evaluation script** (todo). `scripts/eval/final_eval.py`: rebuild the model from a checkpoint (kernel or
   master mode), score every non-overlapping 2048-token window of `wiki32k_val.bin` (~975 windows, 2.0M tokens) and the
   first 2000 windows of `fwedu32k_val.bin`, save per-window losses (`.npy`) + mean, bits per byte. Same windows for
   every model, independent of the training seed. Test on one checkpoint on a lab GPU (a 110M eval fits beside a run).
3. **E2 Paired comparison** (todo). `scripts/eval/compare.py`: paired bootstrap (10k resamples over windows) of the
   loss difference between two runs, 95% CI; seed-level mean and spread; table of the headline comparisons.
4. **R1 Review: data and evaluation** (todo). `scripts/data/prep_wiki.py` (train/val split by article? any overlap?
   dedup?), `get_batch`, `evaluate`, tokenizer, bits-per-byte constant. Look for leakage and seed effects.
5. **R2 Review: kernels** (todo). `bitnet/kernel.py`: pack/unpack round trip, ternary GEMMs vs a dense reference
   (forward, dx, dw; triton and cublas), activation quantisation, edge shapes. Write `tests/test_kernel.py` and run it
   on a lab GPU.
6. **R3 Review: TTF and hooks** (todo). `ts_layer` / `ts_step` vs FORMULAS.md 2a (bias corrections, leak, lr ratio,
   firing bounds, spend, factored v), `bitnet/flip.py` hooks (fused, gradient accumulation), checkpoint/resume of
   `ts_state`. Any place where the method sees information master does not (or vice versa)?
7. **R4 Review: the master baseline** (todo). `bitnet/master.py`, `master_opt.py`, master path in `train.py`: absmean
   weight scale, per-token 8-bit activations, STE, AdamW (betas, eps, weight decay and what it applies to), schedule,
   float tail treated the same as ours. Is anything handicapping master? List the differences from the BitNet b1.58
   recipe (as a threat to note, not to copy into our method).
8. **P1 Predecessor baseline: Bop-style flips** (todo; the user confirmed 3 Oct 00:25: "prior work baselines", as comparisons only). The closest prior work to TTF is a latent-free flip optimizer
   with one timescale (Bop: EMA of the gradient, flip when it exceeds a threshold). Implement `--bop` for ternary
   (per-weight fp32 EMA m with rate gamma; optional normalisation by the factored second moment; move one level against
   sign(m) when |m| > tau and within bounds; no accumulator, no short momentum). CPU unit test, then a 3-setting sweep at
   110M on machines freed at ~04:30-05:00 (4090 / harlequin / pintail / albacore / elver). Our own predecessors (old rule,
   2.815 / 340M 2.7246 / 1.3B tonight) already have numbers.
9. **M1 Real compressed storage** (todo). Store the TTF state as int8 + per-column fp32 scales and fp16 (instead of
   rounding fp32 copies): `--ts_store`. Same results as the simulated formats (same RNG for stochastic rounding);
   measure the memory saved at 110M / 340M.
10. **M2 State in host memory** (todo). `--ts_offload`: pinned host copy of each layer's state, async copy-in before the
    layer's backward update and copy-out after (side stream, prefetch the next layer). Equivalence test + time per step.
11. **M3 27B on one GPU** (todo). A ~23B preset (width 5120, 83 layers, 8 KV heads of 128, MLP 13824; check GQA support),
    initialised directly as packed trits (no fp32 copy), full activation checkpointing, fused update, M1 + M2. Run a few
    steps at micro-batch 1 x 2048 on a 16 GB 4070 Ti Super (and a 24 GB card): measured peak memory and s/step. This is
    the headline ("27B trains on one 16 GB GPU") - only claim what is measured.
12. **E3 Zero-shot tasks** (todo). Download on the laptop CPU (HF datasets): LAMBADA, HellaSwag, PIQA, ARC-e/ARC-c,
    WinoGrande; `scripts/eval/zeroshot.py` (log-likelihood multiple choice, accuracy and length-normalised accuracy, with
    standard errors); run on the headline checkpoints. Expect near-chance at 110M; report with error bars.
13. **R5 Numbers audit** (todo). Script that recomputes every number in RUN_INDEX.md's headline tables from the
    metrics files and flags mismatches.
14. **T1 THREATS.md** (todo). Every attack we can think of (tuning fairness, seeds / CIs, tokens, float extras, eval
    windows, dataset, scale, wall-clock, memory measured vs estimated, code bugs, the extras in the fair reference), the
    evidence for each, and what is still open.
15. **E4 Run the final evaluation** on every rescued checkpoint (E1 + E2), table into RUNS.md / RUN_INDEX.md.

Monitoring (every :17 slot first): `bash $CLAUDE_JOB_DIR/tmp/status_oct3.sh`; finals into RUNS.md / RUN_INDEX.md /
QUEUE.md; crashed runs resumed, nothing new launched beyond these items.
