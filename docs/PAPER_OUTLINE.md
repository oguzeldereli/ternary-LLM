# Paper outline (draft, 2 Oct; local only, not pushed)

Working title options:
- *Two-Timescale Threshold Flips: Training Ternary LLMs Without Master Weights*
- *What Master Weights Actually Do, and How to Train Ternary Networks Without Them*

## Abstract (draft)

Ternary (1.58-bit) language models are usually trained with a full-precision copy of every weight (master weights) and
Adam's state: ~16 bytes per parameter. We ask which parts of that update a ternary network actually needs. Removing one
ingredient at a time from master-weight training shows it needs (i) a short momentum (~10 steps) for the direction, (ii)
a per-weight integrator with a ~1000-step memory that decides when a weight changes, and (iii) an immediate change when
the integrator crosses a boundary; it does not need a full-rank first moment or a per-weight second moment. We build a
master-free rule with exactly these parts, kept low-rank: a short low-rank momentum, Adam-normalised, feeds a long
low-rank accumulator, and a ternary weight flips the step its accumulated evidence crosses a threshold (two-timescale
threshold flips, TTF). With [N] bytes of optimizer state per parameter, TTF matches or beats master weights at 110M,
340M and 1.3B parameters ([numbers]), including against master given the same auxiliary float parameters.

## 1. Introduction
- Ternary LLMs: cheap inference, but training still needs master weights: memory is the barrier to training large ternary
  models on small hardware.
- Contributions: (1) an ablation of master-weight training that isolates what it needs; (2) TTF, a master-free rule with
  sublinear state built on those findings; (3) results at 110M-1.3B (and goldbug 1.3B on FineWeb-Edu); (4) a memory
  analysis (~1 B/param at 27B) and storage formats (8-bit direction, ~10-bit accumulator).

## 2. Setup
- Model: Llama-style transformer, ternary {-1,0,1} linears (BitNet b1.58-style absmean for master; packed 1.6-bit trits for
  ours), 8-bit activations; float tail (embedding, norms) with AdamW. Sizes: 110M (small), 340M (d1024_l24), 1.3B
  (d2048_l24).
- Data: English Wikipedia (Nov 2023), Llama-2 32k tokenizer, 300M tokens, 32k tokens per step, cosine schedule; FineWeb-Edu
  for goldbug. Metric: validation loss (nats/token) and bits per byte (3.58 bytes/token).
- References: full precision (2.683), master weights (2.7513), master + our float extras (2.730, two seeds).

## 3. What master weights need (the ablation)
- Method: `bitnet/master_opt.py` (`--m_*`), each from scratch, one change.
- Table 1 (110M): factored v 2.7513; rank-512 first moment 2.7710; + gate 2.7325; leak tau 1000 2.7298; **snap 2.8225
  (= our earlier rule)**; **leak tau 300 2.8479**; **beta1 0.997 3.2446** (beta2 0.999 alone 2.7685); **rate-limited firing
  3.5214**.
- Figure: night_oct2.png panel 1 (gap to master over training).
- Reading: two timescales + immediate firing; boundary residence (the leftover offset) matters; rank and per-weight
  normalisation do not. Churn statistics (96% of master's late changes are reversals) as supporting analysis.

## 4. Two-timescale threshold flips (TTF)
- Equations (FORMULAS.md 2a): factored v; short momentum m (rank r_s, beta 0.9) by subspace iteration; u = -m/sqrt(v);
  accumulator A (rank r, leak tau) += u * (lr ratio)^p; flip where |A| >= theta, spend theta; low-rank spend U -= theta D V.
- Design choices backed by ablations: theta 16 (8 / 24 / 32 worse); tau 1000 (300 worse, 3000 similar); annealing p = 0.5
  holds the late slope; spend theta (2 theta worse); gate neutral.
- Implementation: per-layer update fused into the backward (exact; frees the gradient buffer), Cholesky-QR, cuBLAS path.

## 5. Results
- Table 2 (110M, two seeds): TTF ranks 512 2.702 (2.7067 / 2.6978); full rank 2.692; full rank + annealed leak 2.670;
  vs master + extras 2.730, master 2.7513, earlier rule 2.815.
- Rank (Table 3): 512/512, 512/256, 256/512, 256/256, 128/128, 64/64 (2.7067 ... 3.0183); both ranks matter equally.
- Scale (Table 4): 340M TTF 2.6268 vs master 2.6626; 1.3B TTF [final, Sat 3 Oct] vs master 2.5868; goldbug 1.3B
  FineWeb-Edu [8-10 Oct].
- Figures: best_runs.png, runs_1b.png, gap-to-master by size.
- Late-phase behaviour: slopes (Sec. 13:10 entry in RUNS.md) and the annealing fix.

## 6. Memory
- Bytes per parameter (27B budget): trits 0.2, float tail 0.08, state 0.66 (int8, ranks 1024) -> 0.94 (master 16).
- Storage formats (Table 5): direction int8 / fp8 free; accumulator needs ~10 bits (fp16 / bf16); int8 accumulator costs
  0.02-0.03 (stochastic rounding noise integrates over the memory; nearest rounding costs less); diagnostics (crest ~4,
  range 15-17 bits within a matrix).
- Measured persistent memory at 1.3B: 5.3 GiB (TTF) vs 19.4 GiB (master); peaks and how to reduce them (fused update,
  checkpoint every layer).

## 7. Analysis (optional / appendix)
- Where the gap was (rare word pairs, context position) for the earlier rule; why the earlier rule's flips undo master's
  state (branch experiments); cheap-selection negative result.

## 8. Limitations
- Scale tested up to 1.3B (goldbug); one corpus at 300M tokens for most ablations; single seeds at 340M / 1.3B; float
  extras; wall-clock (our kernels vs cuBLAS); rank needed at large width unknown.

## Related work
- To be written after the professor has seen the results (kept raw until then).

## To do before submission
1. 1.3B TTF final (Sat 3 Oct) and goldbug results (8-10 Oct).
2. Fair reference at 340M (master + extras) if time.
3. Storage-format finals (2 Oct night); one full run with the chosen split.
4. Speed numbers after the cuBLAS / Cholesky fixes.
5. arXiv (cs.LG; endorsement from a cs.LG endorser).
