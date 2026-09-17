# ternary-LLM

From-scratch **native ternary** ({-1, 0, 1}) language models — BitNet-style, but
trained **without full-precision master weights**, with custom Triton kernels that
keep the weights packed at **1.58 bit** (5 trits/byte) even during training.

Two things live here:
- **A systems result:** a 27B ternary model *trains* on a single 12 GB consumer
  GPU (172 tok/s, ~9.9 GiB) — weights never materialized dense.
- **A study:** how far can you get *without* the master weights, and does a cheap
  per-weight accumulator recover the gap. See **[RESULTS.md](RESULTS.md)** and
  **[NOTES.md](NOTES.md)**.

## Headline numbers

27B training on one 12 GB GPU (seq 512):

| | tok/s | peak VRAM |
|---|---|---|
| torch unpack+linear | 67 | 10.86 GiB |
| + Triton kernels (fwd, grad_x, fused-flip) | **172** | **9.93 GiB** |
| + int8 tensor-core forward, norms inside checkpoint | **170** | **8.55 GiB** |

Master-free study, 110M params, seq 2048, 32,768 tokens/step, 300M tokens of
Wikipedia, everything identical except the weight-update rule:

| training mode | per-weight state | perplexity |
|---|---|---|
| latent master + STE + AdamW (**ceiling**) | 12 B/param | **15.79** |
| stateless flips, annealed rate | 1.58 bit | **98.56** |
| stateless flips, annealed rate, 600M tokens | 1.58 bit | **92.72** |
| stateless flips, constant rate | 1.58 bit | 135.84 |
| stateless flips, frozen absolute threshold | 1.58 bit | 160.68 |

Master-free ternary costs **6.2x perplexity** against its own ceiling. The plateau
that earlier versions of this repo reported at 1552 ppl was two bugs (no weight
scale in the forward, and a bf16 tail that froze every norm gain at exactly 1.0)
plus a flip rule that never stops flipping — see [RESULTS.md](RESULTS.md) §0 and
[NOTES.md](NOTES.md).

## Training modes (`--mode`)

| mode | what | state per weight |
|------|------|------------------|
| `kernel` | pure-ternary, stochastic flips, Triton kernels (fastest) | trit only |
| `master` | latent weights + STE + AdamW — the `bit -> inf` ceiling | fp32 latent + Adam |
| `evidence` | `kernel` + k-bit saturating counter (`--ev_bits`) | trit + k bits |
| `stateless` | pure-ternary flips, torch path | trit only |
| `flip` | learned flip-predictor + int8 evidence (torch) | trit + int8 |
| `latent` | int8 latent master-free path | int8 |

Key flags for the flip rule: `--rate_schedule {const,cosine,linear,exp}` (annealing
the flip rate is the single biggest quality win), `--rate_min` (non-zero floor),
`--abs_scale` (freeze the threshold denominator — measured *worse*),
`--err_feedback --ef_alpha` (push unapplied flip residual into grad_x — measured
*worse*: 156 vs 98.6), `--int8`
(s8 tensor-core forward), `--track_flips` (per-layer flip-rate telemetry).

## Quickstart

```bash
pip install -r requirements.txt
python3 test_flip.py                                   # sanity

# data: your own text, or Wikipedia
python3 prep_wiki.py 1.25e9                             # ~1.25B GPT-2 BPE tokens
# or: python3 -m bitnet.data_prep_words --input corpus.jsonl   # word-level

# train (kernel mode; add --max_temp N for the thermal guard)
python3 -m bitnet.train --preset s50 --mode evidence --ev_bits 3 \
    --data data/wiki_train.bin --val data/wiki_val.bin \
    --seq_len 512 --batch_size 16 --lr 3e-4 --out_dir checkpoints/run

# resume anytime
python3 -m bitnet.train ... --out_dir checkpoints/run --resume

# generate from a (mid-training) checkpoint
python3 generate.py --ckpt checkpoints/run/ckpt.pt --vocab data/wiki_vocab.json \
    --prompt "the history of" --tokens 80

# deploy: collapse to packed ternary (~1.6 bit)
python3 -m bitnet.pack --ckpt checkpoints/run/ckpt.pt --out deploy.pt
```

Presets (`bitnet/config.py`): `small` 110M · `s50` 51M · `d1024_l24` 340M ·
`b27` 27B · `b40` 37B.

## Features

- **1.58-bit packed weights, resident during training** — 5 trits/byte, unpacked
  only transiently one layer at a time.
- **Triton packed-ternary kernels** (`bitnet/kernel.py`): `tern_gemm` (forward),
  `tern_gemm_dx` (backward), `fused_flip` (in-place stochastic flips), autotuned,
  trits decoded in-register — no dense weight ever built.
- **Master-free training** — the ternary weight is the only stored state; updates
  are discrete flips, optionally gated by a k-bit per-weight counter.
- **bf16-state Adam** (`bitnet/opt8.py`) for the small float tail (embeddings).
- **Resumable + atomic checkpoints + thermal guard** (`--max_temp` pauses on heat).

## Files

| path | role |
|------|------|
| `bitnet/kernel.py` | Triton packed-ternary GEMM (fwd, grad_x, fused flip) |
| `bitnet/flip.py` | flip layers: stateless, k-bit evidence, kernel, predictor |
| `bitnet/packed.py` | 5-trits/byte packing |
| `bitnet/model.py` | Llama-style transformer (RMSNorm, RoPE, SwiGLU, GQA) |
| `bitnet/config.py` | model/train config + presets |
| `bitnet/train.py` | training loop (resume, checkpoint, thermal guard) |
| `bitnet/opt8.py` | bf16-state AdamW for the float tail |
| `bitnet/pack.py` | collapse checkpoint to deployable packed ternary |
| `generate.py` | sample text from a checkpoint |
| `prep_wiki.py`, `bitnet/data_prep*.py` | tokenize data to uint16 streams |
| `superpose_test.py`, `superpose_sweep.py` | the superposition negative result |
| `bitnet/master.py` | latent-master baseline (STE + AdamW) — the ceiling |
| `diag_snr.py` | split-batch gradient SNR per layer |
| `plot_run.py`, `plot_compare.py` | loss / flip-rate / never-flipped plots |
| `gen.py` | sample text from a wiki32k checkpoint |
| `RUNS.md` | every run: config, wall clock, final ppl |

## Honest limitations

- Master-free training **costs 6.2x perplexity** vs a latent-master baseline at
  identical config (98.56 vs 15.79). Annealing the flip rate closed 27% of the
  master-free gap; doubling the token budget to 600M buys only a few points, so the
  rest looks rule-bound rather than data-bound.
- **n = 1, one model size, one dataset, and everything is undertrained** (300M
  tokens for a 110M model is ~7% of Chinchilla). Differences within the annealed
  group are inside the ±5 ppl eval noise. See RESULTS.md §4.
- The **accumulator bit-depth curve is currently unmeasured** — the 2-bit / 3-bit
  evidence runs predate the beta fix and were not re-run.
- 27B *fits* on one GPU but can't be *pretrained* there — 170 tok/s vs the ~540B
  tokens Chinchilla wants is ~100 years. The wall is compute, not memory.
- Ternary gives a real but modest compute win on current GPUs: the int8 tensor-core
  forward is 1.7-2.0x per layer but only **1.12-1.15x end-to-end**, because
  attention, norms and the output head don't touch those kernels.
