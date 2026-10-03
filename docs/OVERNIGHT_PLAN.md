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
| barbel / flounder / goldeye | `mx_lr075_s0` / `mx_lr3e3_s0` / `mx_lr6e3_s0` (lr 6e-3 diverged: val 5.23 at 500, 5.9 from 1000, 6.1 at 4500; stopped 02:30, goldeye free) | "master baseline under-tuned": master lr 7.5e-4 / 3e-3 / 6e-3 (only 1.5e-3 so far) | master 2.7513 | ~05:00 |
| lamprey / quillback | `mx_seed2` / `mx_seed3` | "one master seed": error bars for the reference | master 2.7513 | ~05:00 |
| hake | `ts16rs512tau1000an05_seed3` | third seed of the headline TTF run | 2.7067 / 2.6978 | ~05:00 |
| koi | `tsbare_rs512an05_s0` | "the float extras do the work": TTF without row/col scales, adapter, qk temperature | master (bare) 2.7513 | ~05:00 |
| tope / rudd | `fw_ts512an05_600M` / `mx_fw_600M` | "advantage vanishes with more tokens / one dataset": FineWeb-Edu, 600M tokens, no repetition | each other | ~10:00 |
| dory / uaru | `big_ts16r256rs256tau1000_s0` / `big_ts16r128rs128tau1000_s0` | rank at width (27B) | 340M r512 2.6268, master 2.6626 | Sat evening |
| albacore / elver | `qm8a16_seed2` / `qmf8a16_seed2` | storage "free" beyond noise | fp32 seed 2 2.6978 | ~04:50 |
| 4090 / harlequin / pintail | `qfp16s_s0` / `qm8a16v8_s0` / `qm8a8det_s0` | storage formats | see QUEUE.md | ~04:30 |
| inanga | `orth_debug.sh` | Cholesky-QR | Householder | done: `chol` / `chol_cs` broken (zero-state lock), **`chol64` = Householder** (5.157 vs 5.162 at 300, orthogonality 7e-7, no zero inputs, input condition up to 5e6); second pass with zero counters running |

Note: the training-time eval windows are seeded by the run's seed (`tc.seed + 12345` in `evaluate`), so seed-2/3
runs are scored on different val windows from seed-0 runs. Paper numbers must come from the final evaluation (E1).

## Work items (in order)

Status: todo / in progress / done (with a short result).

1. **C1 Checkpoint inventory and rescue** (done, 01:30: 55 lab + 8 4090 runs, 25 GB in `checkpoints/final/`). Inventory of all lab PCs in
   `scripts/eval/inventory_lab_2026-10-03.txt` (172 run dirs; ~95 finished). The final checkpoint is `ckpt.pt`
   (`save_ckpt(end_step - 1)`, step 9154). `scripts/eval/rescue.sh LIST` writes `eval.pt` on the remote machine (ckpt
   without optimizer, TTF state and flip masks: 164 MB per 110M TTF run) and copies it with train.log and metrics.jsonl
   to `checkpoints/final/RUN/` (full `ckpt.pt` too for the headline runs). Lists: `scripts/eval/rescue_lab.list` (55
   headline lab runs: TTF, master references, old rule, rank sweep, storage formats, design ablations) and
   `rescue_4090.list` (340M master / TTF / old rule, `ts16full_tau1000_s0`, `qabf_s0`, `mx_lag02`). Myriad finals
   (`mx_factv`, `mx_leak300`, `mx_rank512`, `mx_snap`, `ours_vfull_r1024`, `x1b_master_lr75`; Scratch persists) stay
   there and are evaluated there. Already local: `master_tracked` (ckpt_9154.pt). Not found anywhere: none of the
   headline runs is missing.
2. **E1 Final evaluation script** (done, 01:40). `scripts/eval/final_eval.py RUN_DIR...` rebuilds the model from the
   checkpoint alone (mode, config, layer options saved in it; extras switched on when their parameters are in the state
   dict; strict load), scores all 974 non-overlapping 2048-token windows of `wiki32k_val.bin` and the first 2000 of
   `fwedu32k_val.bin`, and writes per-window and per-position losses (`final_eval.npz`) + `final_eval.json`. Self-check:
   it recomputes the training-time eval with the run's seed and compares with the logged FINAL: TTF r512 2.70673 vs
   2.7067 (+3e-5), old rule seed 2 (-2e-5), master + extras (+5e-5), so the rebuilt models are the trained ones. First
   full-val numbers (wiki, 974 windows): TTF r512 **2.6903** (bpb 1.084), master + extras **2.7151** (1.094), old rule
   seed 2 2.7973; FineWeb-Edu (2000 windows) master + extras 3.6837. 75-210 s per model on a lab GPU beside a running
   training. FineWeb val is only on tope / rudd / zander (copy `fwedu32k_val.bin`, 20 MB, before E4).
3. **E2 Paired comparison** (done, 01:50). `scripts/eval/compare.py A B` (or `--groups "NAME=d1,d2" ...` for seed
   means and spread): paired block bootstrap over windows (blocks of 4 consecutive windows, 10k resamples). First real
   result (wiki, 974 windows): TTF ranks 512 seed 0 vs master + extras seed 0 **-0.0249, 95% [-0.0265, -0.0232]**, 86%
   of windows favour TTF; TTF vs old rule seed 2 -0.107. The window interval covers evaluation noise only; seed spread
   comes from the seed runs (E4).
4. **R1 Review: data and evaluation** (done, 02:25).
   - Wikipedia (`scripts/data/prep_wiki.py`): 20231101.en streamed in dataset order, Llama-2 32k tokenizer
     (`hf-internal-testing/llama-tokenizer`), EOT `</s>` between articles; val = the last ~2M tokens, cut at an article
     boundary (1,749 articles, 1,995,459 tokens; train 398.5M). Leakage check (`scripts/analysis/val_overlap.py`): **no
     val article occurs exactly in train**; at most **1.58%** of val 32-token sequences occur anywhere in train (rolling
     hash, upper bound; boilerplate), the same for every model, so no bias between methods.
   - FineWeb-Edu (`prep_fineweb.py`, sample-10BT, same tokenizer): val = the first 10M tokens' documents, written before
     train, disjoint by document.
   - Bytes per token re-measured (decode of the val sets, EOT counted as a token): Wikipedia **3.569** (docs used 3.58:
     bits per byte were 0.3% low, e.g. TTF 1.084 -> 1.087), FineWeb-Edu **4.003**; `final_eval.py` updated, now also
     reports FineWeb bits per byte.
   - Sampling: `get_batch` draws uniform random windows (with replacement) from train with a generator seeded by the
     run's seed, so runs with the same seed see the same batches (TTF and master included); 300M samples from 398.5M
     tokens. `evaluate` uses 30 fixed batches seeded by seed + 12345: same windows for all seed-1337 runs, different
     windows for seed 2 / 3 runs (training-time numbers of different seeds are not comparable; use `final_eval.py`).
   - Threat for T1: hyperparameters of TTF (theta, tau, ranks, annealing) were chosen on this Wikipedia val set (the
     training-time windows are about half of it), while master's lr was never tuned. Answers: the master lr sweep
     running now; report FineWeb-Edu val (never used for any choice) as the untouched test set next to Wikipedia.
5. **R2 Review: kernels** (done, 02:55: **all 69 checks pass** on goldeye, log `tests/test_kernel_ref_goldeye_2026-10-03.txt`;
   max relative errors: int8 forward 3.1e-3, bf16 forward 3.4e-3, input gradient bf16 ~3e-3, int8 paths 0.8-1.6e-2 (the
   int8 rounding of the gradient, unused by the runs), layer forward / straight-through input gradient 3.5e-3, captured
   weight gradient exact; K = 1000 / 2049 (padding) correct). `tests/test_kernel_ref.py` (pack/unpack incl. K not a multiple of 5,
   trit_beta, int8 / bf16 forward GEMMs, input gradient bf16 / int8, weight gradient int8 / cuBLAS, both backends; the
   layer's forward, straight-through input gradient and the captured weight gradient against beta * Q8(x) @ W^T). Notes:
   - Configuration of all runs: `--int8 --dw_mode dense`, `int8_dx` off: forward int8 x trits exact in int32 (per-token
     absmax 8-bit activations), input gradient bf16 (`tern_gemm_dx`), weight gradient dense `gy^T @ (xq * xs)` in the
     autocast dtype; the 8-bit gradient kernels (`dw_int8`, `dw_cublas`, `tern_gemm_dx_i8`) are not used by any
     reported run.
   - Output scale: `beta = 1 / sqrt(K rho)` from the trit density (no learnable scale). Master's absmean gamma follows
     its latent weights, which AdamW can grow or shrink, so bare master has a learnable per-tensor scale that bare TTF
     does not. The bare-TTF run (+0.119 at 4000) therefore also tests "no scale freedom at all"; a fair minimal
     version would give each layer one learnable scalar (for T1; not launched).
   - Padding trits past K are packed as -1 and excluded from beta; whether the GEMMs mask them is what the K=1000 /
     2049 cases test.
6. **R3 Review: TTF and hooks** (done, 03:10). `ts_layer` matches FORMULAS 2a: factored v with one bias correction
   (R_i C_j / mean R carries the factor once), the short momentum and the accumulator are one subspace-iteration step of
   b1 M + (1 - b1) g and (1 - lambda / tau) A + u (V_new = orth(M^T U), U_new = M V_new, the stored matrix is M projected
   on span V_new), m bias-corrected, firing at |A| >= theta within [-1, 1], spend projected (U -= c theta D V). Hooks:
   with `--ts_fused` each layer updates after its input gradient is computed, and checkpointed blocks recompute before
   their own update; gradient accumulation sums micro-batches (u = m / sqrt(v) is scale-free, so sum or mean does not
   matter). The full TTF state (t, v, m, U, V) is checkpointed; resume was tested. TTF sees nothing master does not
   (same batches, no validation data, no look-ahead).
   Findings: (a) doc fix: with `--ts_tau_anneal` the leak scales by rho^p, not rho (FORMULAS corrected; the 1.3B run uses
   rho^0.5 for both); (b) no reported run used `--ts_orth chol` (lab queue log, Myriad and 4090 scripts), so the
   Cholesky bug touches only speed measurements; (c) for T1: master clips the global gradient norm (latents + float
   tail) at 1.0, TTF clips only the float tail and feeds the raw ternary gradients into m / sqrt(v) (scale-free, so
   clipping would only change spikes); (d) bare TTF has no learnable output scale (R2).
7. **R4 Review: the master baseline** (done, 03:35). Master is the BitNet b1.58 straight-through recipe: fp32 latent
   (bf16 would round AdamW steps away), absmean gamma per tensor, round-and-clamp to {-1, 0, 1} * gamma, per-token 8-bit
   absmax activations with a straight-through estimator, bf16 autocast GEMMs like ours, AdamW (0.9, 0.95), weight decay
   0.1 on latents and embeddings and 0 on norm gains, cosine schedule with 305 warmup steps, global clip 1.0; the
   evaluated network is exactly ternary. The float tail is treated identically in both modes (fp32 AdamW, same groups,
   weight decay and learning rates; row / column scales and attention temperature at weight decay 0, adapter at
   `--mag_wd`), so "master + extras" carries exactly our extras. Learning rate: of 7.5e-4 / 1.5e-3 / 3e-3 / 6e-3, 1.5e-3 is
   best so far (+0.015 / 0 / +0.080 / diverged at ~7000), so the reference was not under-tuned in lr (finals ~05:00).
   Differences from the published BitNet b1.58 training recipe, for T1 (not to be copied into TTF): their two-stage
   learning rate (a drop half-way) and weight-decay schedule (off in the second stage) versus our plain cosine and
   constant decay; their squared-ReLU / SubLN blocks versus our Llama-style SwiGLU with pre-RMSNorm (the same for both
   methods here). A master run with the two-stage schedule would answer "a stronger master recipe exists".
8. **P1 Predecessor baseline: Bop-style flips** (in progress, 04:20; the user confirmed 3 Oct 00:25: "prior work
   baselines", as comparisons only). `--bop` in `train.py` (`bop_step`, separate from `--ts`; needs `--lowrank 1` only for
   the gradient capture): per-weight fp32 EMA m <- (1 - gamma) m + gamma g, test |m / sqrt(v)| > tau with the factored
   second moment (Bop2ndOrder-style normalisation; `--bop_raw` for the original raw-m test), move one level against
   sign(m) within [-1, 1]; as in Bop no bias correction (`--bop_bc` to add it), no reset after a move, no short momentum,
   no rank limit, no schedule. CPU test: consistent gradients drive rows one level per step to the bound, noise rows
   mostly stay. Calibration: at equilibrium the noise of m / sqrt(v) is ~ sqrt(gamma / 2) = 0.022 at gamma 1e-3 (TTF's
   theta 16 over a 1000-step memory would correspond to 0.016, inside the noise; TTF survives this because it spends
   theta on every move, Bop does not). Sweep at 110M with the same float extras as TTF (compare TTF 2.7067, master +
   extras 2.730): tau 0.02 / 0.05 / 0.15 (~1 / 2 / 7 sigma of the noise) on goldeye / inanga / 4090, started 04:20, ETA
   ~09:00. Finding for the write-up: TTF = this normalised one-timescale rule + short momentum + spend on firing +
   low rank + schedule scaling. Existing TTF ablations cover part of the difference (tau 300, spend 2, short-momentum
   rank); the one that isolates Bop's missing piece is TTF with `--ts_spend 0` (no reset after a move): proposed for
   the morning, not launched.
9. **M1 Real compressed storage** (done, 05:40). `--ts_store_m FMT --ts_store_a FMT` (int8 / int8det / fp8 / fp16 /
   fp16s / bf16): `ts_layer` keeps the state as codes + per-column fp32 scales (int8, fp8, fp16s) or 16-bit tensors (flat
   tuples, so checkpoint and resume code is unchanged) and decodes on read; `--ts` logs the stored bytes at steps 10 and
   1000. CPU: decoded values bitwise identical to the simulated `--ts_qfmt*` formats for every format. GPU
   (`scripts/lab/store_test.sh`, quillback, 110M ranks 512, 200 steps): **stored state 684.7 MiB (fp32) -> 257.5 MiB
   (direction int8 + accumulator fp16, 0.376x as predicted)**, peak 7.68 -> 7.26 GiB; losses fp32 5.6584 / simulated
   5.6598 / real 5.6576 (identical runs differ by ~0.002 on the GPU, e.g. Householder 5.1622 vs 5.1601, so equality is
   checked on the CPU). At 110M a rank of 512 is close to the layer width, so the low-rank state is not small here
   (8 bytes per ternary weight in fp32, 3 with int8 / fp16); it is sublinear only when width >> rank (27B: M3).
10. **M2 State in host memory** (done, 06:40). `--ts_offload`: every state tuple of `ts_layer` (accumulator, short
    momentum, factored v; encoded when `--ts_store_*` is on) lives in pinned host buffers (allocated once, reused); the
    layer's update copies it to the GPU (`non_blocking`, current stream) and back after the update, so ordering holds
    without events or host syncs. Test (quillback, 110M ranks 512, 200 steps; `scripts/lab/offload_test.sh`): int8 / fp16
    on the GPU 5.6587, peak 7.26 GiB, 433.9 s; **the same offloaded 5.6593, peak 7.01 GiB, 443.8 s (+2%)**; fp32
    offloaded 5.6574, peak 7.01 GiB, 456.2 s. The copies do not overlap compute yet (side stream + prefetch of the next
    layer in backward order would hide them); at 110M PCIe time is small, at 27B it is ~43 GiB per step read + write
    (~1.7 s at 25 GB/s). Lab PCs have 30 GB RAM (28 available): a 27B int8 / fp16 state (~21 GiB) fits pinned.
11. **M3 27B on one GPU** (in progress, 07:15). Preset `b27` (width 5120, 83 layers, 40 heads / 8 KV heads of 128, MLP
    13824, 32k vocab: 22.85B ternary + ~0.29B float). Ternary layers are initialised one at a time on the CPU (a 283 MB
    fp32 temporary per layer, packed at once), so no full-precision copy exists. `scripts/lab/b27_smoke.sh` on barbel
    (4070 Ti Super 16 GB, 30 GB RAM): TTF with `--ts_fused --ts_store_m int8 --ts_store_a fp16 --ts_offload`,
    `--ckpt_skip 0` (every layer checkpointed), fp32 float tail with AdamW, micro-batch 1 x 2048, 5 steps at ranks 256
    and then 1024; TERN_TIME step times and GPU memory, nvidia-smi peak, host max RSS. Estimate for the GPU: packed trits
    4.3 GiB + fp32 tail with AdamW ~4.3 GiB + checkpoints ~1.7 GiB + one block's recompute ~1 GiB + the largest layer's
    update temporaries ~2.3 GiB: ~14 GiB, tight on 16 GB (fallback: bf16 tail + 8-bit Adam, -2.7 GiB).
    **Result, ranks 256 (07:19): the 26.5B-parameter preset (22.85B ternary) trains on ONE 16 GB 4070 Ti Super:** 5 TTF
    steps, 24.6 s per step of 1 x 2048 tokens (forward + backward with the fused update inside; first step 309 s with
    Triton autotuning), torch peak 14.10 GiB allocated (8.61 GiB persistent), nvidia-smi peak 15.95 of 16 GiB (the
    allocator logged mapping retries at the edge; every step completed), host max RSS 13.7 GiB. Ranks 1024 running
    (host state ~21 GiB pinned on a 30 GB machine).
    **Ranks 1024 (07:25): barbel stopped responding** (ssh times out; quillback fine): the ~21 GiB pinned state plus the
    process's own ~13.7 GiB exceeds 30 GB, and pinned pages cannot swap. My estimate missed the process RSS. Needs a
    host with >= 48 GB RAM, or ranks <= 512 (~10.7 GiB host state). Barbel is polled for recovery.
12. **E3 Zero-shot tasks** (done for seed 1337 of three models, 08:30). Datasets (HF, laptop CPU) in `data/zeroshot/*.jsonl`: LAMBADA OpenAI
    5153, HellaSwag val 10042, PIQA val 1838 (`baber/piqa`, the original needs a loading script), ARC-Easy test 2376,
    ARC-Challenge test 1172, WinoGrande-xl val 1267; tokenised with the training tokenizer by
    `scripts/eval/zeroshot_tok.py` (continuation = enc(ctx + cont) minus enc(ctx), EOT first, byte lengths kept).
    `scripts/eval/zeroshot.py`: summed log-probabilities, acc and byte-normalised acc_norm, LAMBADA greedy accuracy and
    target perplexity, WinoGrande partial scoring, standard errors, per-example correctness saved for paired tests.
    Results (110M, 300M tokens; acc / acc_norm; TTF, master + extras, master): LAMBADA 0.135 / 0.123 / 0.126 (target ppl
    51.8 / 59.0 / 57.2); HellaSwag 0.265 / 0.262 / 0.264; PIQA 0.539 / 0.526 / 0.526; ARC-e 0.327 / 0.318 / 0.328; ARC-c
    0.189 / 0.177 / 0.195; WinoGrande 0.500 / 0.493 / 0.490 (chance 0.25 / 0.5 / 0.25 / 0.25 / 0.5). Paired bootstrap over
    examples: LAMBADA TTF - master + extras **+0.012 [+0.005, +0.019]**, TTF - master +0.009 [+0.002, +0.016]; the
    multiple-choice tasks are near chance at this scale and every other difference's interval contains 0 except PIQA
    acc_norm vs master (+0.026 [+0.009, +0.043], one of 20 comparisons). To do: the other seeds, 340M and 1.3B.
13. **R5 Numbers audit** (done, 09:20). `scripts/analysis/audit_numbers.py` checks every RUN_INDEX.md row that names runs
    and gives 4-decimal values against the logged final: 130 match; 5 flagged: three stale "running" rows that still
    showed an intermediate value (cheapcos512 2.8884 at 8250 -> final 2.8634, drywarm512 3.2321 at 3000 -> 2.8199, undo512
    3.3736 at 2250 -> 2.8360; fixed) and two 1e-4 rounding differences (grav05 rows; fixed); 5 runs have no local log
    (the Myriad master ablations mx_factv / mx_rank512 / mx_snap / mx_leak300 and gvsharp_slowgate_s0). No headline number
    was wrong.
14. **T1 THREATS.md** (done, 07:45, moved ahead of E3 / R5): `docs/THREATS.md`, 13 attacks answered with evidence
    (seeds, windows, test-set tuning, leakage, master lr, master implementation, extras, kernels, information, bugs,
    storage, memory, Bop, old rule) and 9 open (late-training shrink at scale, bare comparison, BitNet's two-stage
    schedule, clipping, scale / tokens, wall-clock, rank at width, ranks 1024 host RAM, zero-shot).
15. **E4 Run the final evaluation** on every rescued checkpoint (E1 + E2), table into RUNS.md / RUN_INDEX.md. (Started
    04:30 for the headline groups, moved up because the seed runs finished: TTF -0.028 [-0.029, -0.027] vs master +
    extras on Wikipedia, -0.039 on FineWeb; master seeds 2.7342 / 2.7352 / 2.7383, so the training-time "seed 3 -0.065"
    was the eval windows. 06:40: TTF seed 3 2.6893 (3 seeds: 2.6888, spread 0.0018), storage seed 2 int8 / fp16 2.6877, fp8 /
    fp16 2.6874 (fp32 2.6868), bare TTF 2.8070. To do: storage seed 1337, rank sweep, ablations, 340M, 1.3B on Myriad.)

Monitoring (every :17 slot first): `bash $CLAUDE_JOB_DIR/tmp/status_oct3.sh`; finals into RUNS.md / RUN_INDEX.md /
QUEUE.md; crashed runs resumed, nothing new launched beyond these items.
