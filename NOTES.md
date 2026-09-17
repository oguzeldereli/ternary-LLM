# Notes: why master-free ternary training plateaus, and what actually fixes it

**This file was rewritten after the Phase 0 audit.** The earlier version explained
the plateau as a gradient-noise floor. That explanation is wrong, and the
measurements that refute it are below.

## The setup
Ternary weights `w ∈ {-1,0,1}`, no full-precision master. Each step, per weight, a
stochastic *flip*: probability of moving one level toward `−sign(gradient)`, scaled
by gradient magnitude:

```
gn   = g / mean|g|                      # per tensor, recomputed every step
prob = min(|gn| / g_ref, 1) * rate      # g_ref = 3, rate = 2e-2
```

## What the old explanation said
That the minibatch gradient near an optimum is mostly noise, so flips fire on noise
in random directions, and the weight becomes a random walk that never locks. More
accumulator bits = longer averaging = lower floor.

## Why that is wrong

**1. The gradient is not noise-dominated at this batch size.** Splitting a batch in
half and comparing the two weight gradients (`diag_snr.py`) on a plateaued 110M
model at 32,768 tokens/step:

| | mean sign agreement | mean cosine | implied full-batch SNR |
|---|---|---|---|
| 110M, after the beta fix | **65.7%** | 0.535 | 1.52 |
| 51M, before the beta fix | 54.0% | 0.135 | 0.56 |

50% is pure noise. At 65.7% the flip direction is right about two times in three,
and the strongest 1% of entries agree 93–100% of the time. The model still plateaus.

**2. More averaging does not help.** Quadrupling the batch (32,768 → 131,072 tokens
per step, flips accumulated and applied once per step) did not improve the floor —
it needed 33% *more* tokens to reach the same place. Spatial averaging is not the
missing ingredient.

**3. The flip rate is invariant, and that is the real problem.** `prob` is
normalized by the tensor's *own* mean |g|, recomputed every step. If every gradient
in a layer shrinks as the model converges, the ratio is unchanged and **the same
fraction of weights keeps flipping forever**. Measured: 0.405% of weights flipped
per step at step 0 and 0.420% at step 9155, with never-flipped at 0.02%. The rate
did not move across batch sizes, step counts, token budgets, or learning rates.

The rule has no absolute scale and no way to know a weight is already correct, so a
converged weight is re-flipped at the same rate as a wrong one. The plateau is that
churn, not gradient noise.

## What fixes it
Anneal the flip rate toward zero. One schedule on `rate`, nothing else changed:

| flip-rate schedule | val ppl (110M, 300M tokens) |
|---|---|
| constant 2e-2 (the old recipe) | 135.84 |
| cosine 2e-2 → 0 | **98.56** |
| linear 2e-2 → 0 | 101.38 |
| cosine 2e-2 → 0.02% floor (never zero) | 103.96 |
| frozen absolute threshold, constant rate | 160.68 |

The ordering is monotone in **how much flipping survives late in training**. Every
flip you allow near convergence costs perplexity, and driving the rate to exactly
zero beats leaving a floor.

## What does not fix it

**A frozen absolute threshold (the obvious closed-loop fix).** Calibrate mean|g| per
layer over the first 200 steps, freeze it, and let the flip rate fall on its own as
gradients shrink. It gets *worse* (160.68): gradient magnitude **grows** relative to
its value at init rather than shrinking, so the flip rate rose from 0.58% to 0.92%
and the model churned harder. The premise "gradients shrink as you converge, so an
absolute threshold quiets down" is false for this model.

**Spatial error feedback.** Push each layer's unapplied flip residual `(p − fired)·d`
into `grad_x` so earlier layers compensate, with no per-weight state. Worse at every
strength tried (156.01 vs 98.56 at α=0.01). Since `E[fired] = p`, the residual is
zero-mean sampling noise; feeding it upstream just adds noise to gradients that are
already only ~1.5 SNR. The damage concentrates where flipping is heavy — the run blew
up early and recovered only as the anneal stopped the flips.

**Weights do not lock, even in the runs that work.** never-flipped ended at 0.173%
in the best run: the gain comes from flipping *less overall*, not from individual
weights settling into place. The old note predicted locking; that is not what
happens.

## What it still costs
Against a latent-master baseline at identical config (110M, seq 2048, 32,768
tokens/step, 300M tokens), master-free ternary is **6.2x worse in perplexity**
(98.56 vs 15.79), and doubling the token budget to 600M buys only a few points.
The remaining gap looks rule-bound, not data-bound.

## Why you can't share the accumulator (unchanged)
Storing N independent evolving scalars needs Θ(N). Measured on the real 3-bit
evidence: 75% dense, 2.66 bits/weight of entropy, so a d < N sketch recovers each
item with crosstalk `√(active/d)` and at d=2²⁰ gives coin-flip sign recovery. See
RESULTS.md §3. This applies to any scheme that compresses per-weight state,
including per-neuron accumulators.
