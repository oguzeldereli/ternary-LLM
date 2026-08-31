# Native ternary BitNet — two master-free training modes

A native ternary LLM ({-1,0,1} weights) trained **without a full-precision
master copy of the weights**. Two mechanisms:

- **`flip` (default)** — weights are *pure trits* {-1,0,1}. A tiny shared
  **flip predictor** reads the backprop signal and predicts each weight's
  discrete transition (down / stay / up). Weights only ever change by a
  predicted whole-level flip. See `bitnet/flip.py`.
- **`latent`** — weights stored as an `int8` latent grid, updated in place with
  stochastic rounding inside a fused backward hook. See `bitnet/bitlinear.py`.

Neither keeps an fp32 master or fp32 optimizer states for the ternary matrices.

## Flip predictor (the `flip` mode)

Per weight, persistent state is only: the trit `weight` (int8 -> packs to
~1.58 bit) and an int8 `evidence` register. Each step:

1. Forward feeds `[weight, evidence]` to the shared predictor -> P(down/stay/up);
   `delta = P(up)-P(down)`. A straight-through path makes the task loss train
   the predictor (grad to `delta` == step * dL/dweight).
2. Backward hook integrates pressure: `evidence += quantize(-grad)`.
3. Flip step: a weight shifts one level when `|evidence|` crosses a threshold
   **and** the predictor agrees on the direction; evidence resets on flip.

The predictor is one small net (~hundreds of params) shared across the whole
model and trained by the task loss. It is the learned rule deciding whether a
given backprop step actually flips a bit.

## `stateless` mode — zero accumulator (weights only)

The literal "don't accumulate at all" option. Only per-weight state is the trit
itself, so the entire trainable state of a 37B model is the ~6.8 GiB of packed
ternary weights — nothing else. There is no evidence register and no predictor.

Instead of averaging gradient noise in a per-weight register, it averages in
**time**: each step, each weight flips one level toward the descent direction
with a small probability that scales with this step's gradient magnitude. A
weight with true gradient ≈ 0 flips rarely and symmetrically, so it stays put on
average; a weight under consistent pressure drifts.

Measured (toy model, tail frozen so only weight flips can move the loss):

| mode | loss |
|------|------|
| no flips (baseline) | 5.568 → 5.568 |
| stateless rate 2e-3 | 5.568 → 4.69 |
| stateless rate 2e-2 | 5.568 → **4.29** |
| stateless rate 1e-1 | 5.568 → 4.49 (rate too high, noise) |

So it genuinely learns with zero accumulator. Tradeoff: slower, rate-sensitive
(too high re-injects noise), and convergence to good perplexity at LLM scale is
unproven. `--rate` tunes it.

**Why you can't go below this:** the accumulator (evidence mode) buys faster,
lower-noise convergence by low-pass filtering the minibatch gradient. Stateless
mode trades that for zero memory by filtering in time instead. Either way the
eliminated thing is the high-precision master weight — the 6.8 GiB of trits is
just the model, at train and deploy alike.

## Why there's normally a "master" — and how this removes it

Gradients are tiny continuous nudges; a weight must accumulate many of them
before it should flip ternary level. That accumulation needs *some* persistent
state, but it does **not** need a separate high-precision copy:

1. Forward builds a **transient** float weight `w_real = latent * row_scale`,
   quantizes it to ternary (absmean, straight-through), so gradients reach
   `w_real` as if the quantizer were identity.
2. `w_real` registers a **backward hook**. When its gradient arrives, the hook
   applies the optimizer step to the `int8` latent **in place** using
   **stochastic rounding** (unbiased sub-integer updates), then `w_real` frees.
3. **Gradient checkpointing** means each layer's `w_real` is rebuilt during that
   layer's own backward and released immediately — so peak float-weight memory
   is one matrix, never the whole model.

Persistent weight memory ≈ **1 byte/param** (int8 latent), +1 byte/param if you
turn on int8 momentum. Compare baseline BitNet: bf16 master (2) + Adam m,v fp32
(8) + grad (2) ≈ **12 byte/param**.

**Honest limit:** you can't hit literal 1.58 bit/param with zero accumulator and
still train a stable LLM — pure accumulator-free ternary is too noisy. The floor
here is the int8 latent. The high-precision master copy is what's eliminated.

## Files

| file | role |
|------|------|
| `bitnet/flip.py`      | flip predictor, stateless + kernel ternary layers |
| `bitnet/kernel.py`    | **Triton packed-ternary GEMM** (decodes trits in-kernel, forward path) |
| `bitnet/opt8.py`      | low-memory bf16-state AdamW for the float tail |
| `bitnet/packed.py`    | 1.6-bit packing (5 trits/byte) for resident + deploy weights |
| `bitnet/bitlinear.py` | int8-latent master-free layer + fused backward-hook optimizer |
| `bitnet/model.py`     | Llama-style transformer (RMSNorm, RoPE, SwiGLU, GQA) |
| `bitnet/config.py`    | model/train config + size presets |
| `bitnet/train.py`     | training loop |
| `bitnet/data_prep.py` | text -> uint16 token stream |
| `bitnet/pack.py`      | collapse checkpoint to packed ternary (~1.6 bit) for deploy |
| `test_flip.py`        | proves pure-ternary weights train by predicted flips |
| `test_smoke.py`       | proves the int8-latent path learns in place |

## Presets (sized for one consumer GPU)

| preset | params | deploy size | notes |
|--------|--------|-------------|-------|
| `small`      | ~130M | ~0.05 GiB | debug the pipeline |
| `d1024_l24`  | ~340M | ~0.12 GiB | default; comfy on 12 GiB |
| `d1536_l24`  | ~730M | ~0.25 GiB | push a 12 GiB card: bs=1 |
| `b40`        | ~37B  | ~8.1 GiB  | 40B target. Deploys <10 GiB. Stateless training ~16 GB (weights packed 1.6-bit resident); fits a 24 GB card, or the 12 GB laptop via RAM offload. Compute (trillions of tokens) is the real wall — needs a cluster. |

Deploy sizes are tiny; the 12 GiB budget is a **training** ceiling. Measured:
340M at bs=1/T=1024 with checkpointing peaks **<1.7 GiB**.

## Run

```bash
pip install -r requirements.txt
python3 test_smoke.py                                 # sanity check

# prepare YOUR data (byte-level, no deps; or --tokenizer <hf-name> for BPE)
python3 -m bitnet.data_prep --input corpus.txt --out_prefix data/data

# train (flip predictor is the default; edit vocab_size if you used BPE)
python3 -m bitnet.train --preset d1024_l24 --mode flip --theta 24 \
    --data data/data_train.bin --val data/data_val.bin \
    --batch_size 4 --grad_accum 16 --seq_len 2048

# deploy: collapse to packed ternary (~1.6 bit)
python3 -m bitnet.pack --ckpt checkpoints/ckpt.pt --out deploy.pt
```

- `--mode kernel` (default, recommended): stateless flips with Triton
  packed-ternary kernels (forward, backward grad_x, and in-place fused flip).
  Fastest and leanest; verified 27B trains on a 12 GiB laptop
  (seq<=512, ~172 tok/s, peak 9.93 GiB). `--rate` tunes flip probability.
- `--mode flip`: pure-ternary weights via the flip predictor + int8 evidence
  (torch path). `--theta` = evidence needed to fire a flip.
- `--mode stateless`: zero-accumulator stochastic flips, torch path (no kernel).
- `--mode latent`: int8-latent master-free path. `--momentum 0/0.9`.

> Note: your GPU currently has a `llama-server` (Qwen3 27B) holding ~10 GiB.
> Stop it, or wait until it's free, before training the full-size preset.
