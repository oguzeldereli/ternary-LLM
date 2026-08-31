# Results

Two lines of work: (1) a **systems** result — training a 27B native-ternary model
on a single 12 GB consumer GPU — and (2) a **study** of *master-free* ternary
training, where the standard full-precision "latent/master" weights are removed.

---

## 1. Systems: 27B ternary training on one 12 GB GPU

Native ternary ({-1,0,1}) transformer, weights packed **1.58-bit (5 trits/byte)**
resident during training. Custom Triton kernels keep the dense weight from ever
being materialized.

| stage | throughput @ seq512 | peak VRAM |
|-------|--------------------|-----------|
| unpack + `F.linear` (torch) | 67 tok/s | 10.86 GiB |
| + forward kernel | 92 tok/s | 10.26 GiB |
| **+ backward-`grad_x` + fused-flip kernels** | **172 tok/s** | **9.93 GiB** |

- 27B (`b27`) fits training in **~9.9 GiB** (seq≤512); resident weights only 4.56 GiB.
- Three autotuned Triton kernels (`bitnet/kernel.py`), all decode trits in-register:
  `tern_gemm` (forward), `tern_gemm_dx` (backward grad_x), `fused_flip` (in-place
  stochastic flip on packed bytes). 2.3–3.1× faster per layer than unpack+linear.
- The wall is **compute, not memory**: 172 tok/s × ~540B Chinchilla tokens ≈ 100 yr
  on one laptop. A single GPU can *hold* 27B ternary; it can't *pretrain* it.

Throughput scales ~1/params: 50M ≈ 40k tok/s, 340M ≈ 4k, 1.3B ≈ 1.8k.

---

## 2. Study: does removing master weights work?

Standard BitNet keeps a full-precision latent weight per parameter (12 B/param of
master + optimizer). We test training with **no master** — the ternary weight is
the only stored state, updated by stochastic *flips*. Question: how much does
dropping the accumulator cost, and can a cheap per-weight counter recover it?

**Setup (identical across runs):** 51M-param model (`s50`, GPT-2 BPE vocab 50257),
English Wikipedia, **1.1B tokens**, seq 512, batch 16, same LR schedule. Metric:
held-out validation loss / perplexity.

| mode | per-weight accumulator | val loss | perplexity |
|------|------------------------|----------|------------|
| stateless | none (0-bit) | 7.347 | 1552 |
| evidence | 2-bit counter (±1) | 7.102 | 1214 |
| evidence | 3-bit counter (±3)* | 6.561 | 707 |

\* 3-bit stopped at 58% of the run, still descending — 6.561 is an upper bound.

### Findings
1. **Master-free ternary plateaus.** Stateless (magnitude-proportional stochastic
   flips) drops fast then flatlines at 7.35 — barely below the unigram baseline.
   It learns token frequencies + a little local context, then stalls.
2. **The plateau is a noise floor.** Near convergence the true gradient ≈ 0 but the
   minibatch gradient is noise; a memoryless flip rule flips on that noise forever.
   The weight becomes a random walk that never locks. (See `NOTES.md`.)
3. **A per-weight accumulator lowers the floor, monotonically with depth.**
   0-bit → 2-bit → 3-bit gives 7.35 → 7.10 → 6.56. The accumulator averages the
   noise over time before committing a discrete flip; more bits = longer averaging.
4. **It's still master-free and cheap.** No fp32 master, no optimizer state on the
   weights. A 2-bit counter is 2 bits/param (vs 12 B for a real master).

### What this is and isn't
- It is **not** "master-free training works" — every variant plateaus far above a
  real LM (~4.0). Removing the master hurts a lot.
- It **is** a clean characterization: *accumulator bit-depth vs quality*, with the
  standard master weight as the b→∞ limit. The interesting knob is bits/param.
- A proper **master baseline** (bf16 latent + STE + AdamW) was not run for time; it
  is the missing b→∞ ceiling.

---

## 3. Negative result: superposition can't shrink the accumulator

Can the per-weight evidence be packed into a shared high-dimensional vector
(VSA / Count-Sketch) to beat the per-weight cost at scale? **No — measured on the
real 3-bit evidence.**

- **Evidence is 75% dense** (19.3M of 25.7M counters nonzero at once). Gradient
  noise keeps the counters off zero, so there is no sparsity to exploit.
- Superposition crosstalk is `√(K/d)` (K = simultaneously-active, d = dimensions).
  At d=2²⁰, load = 18 → sign-recovery **56%** (coin flip), and 95% of zero-weights
  get phantom evidence.
- To recover even 91% you need d=2²⁶ = **67 MB**, vs the exact 3-bit array at
  **9.6 MB**. At the array's memory budget the sketch recovers at **52%**.
- **Entropy floor:** the evidence has 2.66 bits/weight of entropy → 8.5 MB
  incompressible. The 3-bit array (9.6 MB) is already ~13% above it. No
  representation beats entropy; superposition adds crosstalk on top, so it always
  loses for dense data.

Why LLMs superpose 10⁵ features in 4096 dims but this can't: LLM features are
**sparse per instance** (few active per token) and time-multiplexed; the evidence
is **75% active, all the time, persistently**. `capacity ∝ d / active`, and here
active ≈ total. Same math, opposite regime.

---

## Reproduce

```bash
python3 prep_wiki.py 1.25e9                       # ~1.25B GPT-2 BPE tokens
python3 -m bitnet.train --preset s50 --mode kernel   --data data/wiki_train.bin --val data/wiki_val.bin --seq_len 512 --batch_size 16 --lr 3e-4 --out_dir checkpoints/mf     # stateless
python3 -m bitnet.train --preset s50 --mode evidence --ev_bits 2 ... --out_dir checkpoints/ev2   # 2-bit
python3 -m bitnet.train --preset s50 --mode evidence --ev_bits 3 ... --out_dir checkpoints/ev3   # 3-bit
python3 superpose_test.py                         # density + superposition recovery
```
