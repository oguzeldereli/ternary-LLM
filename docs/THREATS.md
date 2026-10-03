# Threats to the TTF results, and the evidence (3 Oct 2026, local only)

Every attack we could think of on the claim "TTF trains ternary networks without master weights, with sublinear
optimizer state, and matches or beats master weights", what answers it, and what is still open. Numbers are from the
final evaluation on fixed windows (`scripts/eval/final_eval.py` + `compare.py`, RUNS.md "3 Oct 04:45" and "06:40")
unless marked "training-time".

## Answered

| attack | evidence |
|---|---|
| **Noise / lucky seed** | 3 TTF seeds 2.6903 / 2.6868 / 2.6893 (spread 0.0018); 3 master seeds 2.7342 / 2.7352 / 2.7383 (0.0021); 2 master + extras seeds (0.0021). TTF - master + extras -0.028, 95% [-0.029, -0.027] over windows; 97% of windows favour TTF. The gap is >10x the seed spread. |
| **Evaluation windows chosen favourably** | Every model is scored on all 974 non-overlapping 2048-token windows of the Wikipedia val set and 2000 of FineWeb-Edu val; the rebuilt models reproduce their logged training-time FINAL to < 6e-5. (The training-time evaluation drew its windows with the run's seed: up to 0.065 of apparent difference between seeds came from that alone; those numbers are not used.) |
| **Tuned on the test set** | TTF's settings were chosen on the Wikipedia val set. FineWeb-Edu val was never used for any choice: there TTF - master + extras is -0.038 [-0.039, -0.037], larger than on Wikipedia. |
| **Train / val leakage** | Article-level split; no val article occurs in train; at most 1.58% of val 32-grams occur anywhere in train (boilerplate), identical for all methods (`scripts/analysis/val_overlap.py`). |
| **Master baseline under-tuned (learning rate)** | Master lr 7.5e-4 / 1.5e-3 / 3e-3 / 6e-3: full-val 2.7608 / 2.7342 / 2.7967 / diverged. 1.5e-3, the one used, is the best. |
| **Master baseline implemented wrongly / handicapped** | Review R4: BitNet b1.58 straight-through recipe (absmean gamma, per-token 8-bit activations, fp32 latent, AdamW 0.9 / 0.95, weight decay 0.1, cosine, global clip 1.0), the network evaluated is exactly ternary; float tail treated identically in both modes. |
| **The float extras do the work** | Master given exactly the same extras (row / column scales, rank-16 additive adapter, attention temperature): 2.7166, still 0.028 behind TTF. Extras help master by 0.019 and TTF by more (bare TTF is 0.12 worse than TTF; see "open" for why bare TTF is not a fair bare comparison). |
| **Kernels compute something else than a ternary network** | `tests/test_kernel_ref.py`: 69 checks against dense references (packing incl. padding, beta, int8 / bf16 GEMMs, input and weight gradients, the layer's forward and backward), max relative error 3.5e-3 (bf16 rounding). |
| **The optimizer sees information master does not** | Review R3: same batches (same data RNG for the same seed), no validation data, no look-ahead; the update uses only the current gradient and its own state. |
| **Implementation bugs in TTF** | Review R3 against FORMULAS 2a (one doc fix: the leak scales by rho^p). The fast Cholesky-QR (`--ts_orth chol`) is broken (88 of 168 subspaces lock at zero), but no reported run used it; Householder has 0 zero states at condition numbers up to 5.6e8; `chol64` is equivalent. |
| **Storage formats only simulated** | `--ts_store_*` stores real int8 codes + scales and fp16: bitwise equal to the simulation on CPU, 684.7 -> 257.5 MiB at 110M, loss unchanged. Seed 2: int8 / fp16 2.6877, fp8 / fp16 2.6874 vs fp32 2.6868. |
| **Memory claims only estimated** | Measured: a 26.5B-parameter model (22.85B ternary) trains with TTF on one 16 GB RTX 4070 Ti Super: 24.6 s per 2048-token step, torch peak 14.1 GiB, nvidia-smi 15.95 GiB, host 13.7 GiB (ranks 256, state offloaded). |
| **A known latent-free optimizer already does this** | Ternary Bop (one timescale, normalised, per-weight state): best of tau 0.02 / 0.05 / 0.15 is 3.2451 (tau 0.15, training-time) vs TTF 2.7067: +0.54. |
| **Our own earlier rule was as good** | Old rule (ranks 512): 2.7973 full-val, +0.109 behind TTF; at 1.3B it finished +0.122 behind master. |

## Open (say so in the paper, or answer before submission)

1. **The lead shrinks late in training at larger scale / more tokens.** 1.3B: -0.058 / -0.042 / -0.032 / -0.019 /
   -0.012 vs master at 3000 / 4000 / 5000 / 6000 / 6500 (on trend it ends near or behind master; final Sat ~14:40).
   FineWeb 600M tokens at 110M: -0.095 at 2000, -0.036 at 12250. 340M ranks 128: level with master at 5000. At 110M,
   300M tokens the lead holds to the end. Levers known from 110M: the step annealing p and the leak. This is the main
   scientific risk.
2. **Bare comparison is not like for like.** Bare TTF has no learnable output scale (beta = 1 / sqrt(K rho)); master's
   absmean gamma follows its latent weights, which AdamW scales. A fair minimal version gives each TTF layer one
   learnable scalar (not run).
3. **Stronger master recipes exist.** BitNet b1.58's two-stage learning rate and weight-decay schedule were not
   used for master (or for TTF). A master run with it answers "you beat a weak master".
4. **Gradient clipping differs.** Master clips the global norm including latents; TTF's ternary gradients are not
   clipped (its step is scale-free). Not expected to matter; untested.
5. **Scale and data.** Largest finished comparisons: 340M (TTF ranks 512 -0.036 vs master, one seed, training-time),
   1.3B running. Token budgets are small (300M / 600M); no run at Chinchilla-optimal tokens for 340M+.
6. **Wall-clock.** At 1.3B on an A100 TTF takes 5.1 s vs master 4.5 s per step with cuBLAS + fused update; with
   Householder QR the update cost must be re-measured (the fast `chol` path is invalid; `chol64` is fine on A100).
7. **Rank at large width.** 340M ranks 256 / 128 trail ranks 512 by 0.05 / 0.13 (training-time). Whether the rank
   needed grows with width (27B memory claims assume ranks 256-1024) is not settled.
8. **Ranks 1024 at 27B** on a 30 GB host: the 21 GiB pinned state plus the process's 13.7 GiB exceeded the RAM and the
   machine stopped responding; needs a host with >= 48 GB or ranks <= 512.
9. **Zero-shot / downstream** evaluation not yet run (expected near chance at 110M).
