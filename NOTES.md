# Notes: why master-free ternary training plateaus, and why the fixes are what they are

## The setup
Ternary weights `w ∈ {-1,0,1}`, no full-precision master. Each step, per weight,
a stochastic *flip*: probability of moving one level toward `−sign(gradient)`,
scaled by gradient magnitude. This is the `stateless` mode.

## Why it plateaus (the noise floor)
The minibatch gradient is `g_t = μ + ε_t`: true gradient `μ` + sampling noise
`ε_t` (mean 0, std σ). Near an optimum — where most weights spend training —
`μ ≈ 0`, so `g_t ≈ ε_t`: the flip fires on **noise**, in a **random** direction.

Expected drift per step ∝ `μ·rate` (tiny). Per-step variance ∝ `rate` (a full ±1
jump). So the weight is a **random walk with tiny drift and large discrete steps**.
Ternary weights have only 3 values — no small adjustment — so a weight that "wants"
0.3 can only *dither* between 0 and 1. That dithering never stops (every step
re-rolls one noisy sample), so the weight **hovers, never locks**. The residual
hovering is irreducible loss: the plateau.

## Why an accumulator fixes it
Integrate `g_t` over ~K steps before committing a flip. Averaging K samples cuts
noise by √K, so the flip decision uses a **denoised** gradient. Two effects:
1. Flips follow the consistent signal, not the noise.
2. Once a weight is right, accumulated evidence stays below threshold (noise
   cancels) → it **stops flipping and locks**.

More accumulator bits = longer effective averaging = lower floor. Measured:
0-bit 7.35 → 2-bit 7.10 → 3-bit 6.56. The standard full-precision master weight is
just the `b → ∞` limit of this counter.

"Magnitude-proportional flip probability" (which stateless already does) doesn't
help: magnitude of *one* noisy sample can't tell a small-consistent gradient
(real) from a large-random one (noise). Only averaging over time can.

## Why you can't share the accumulator (superposition / sketches)
Storing N independent evolving scalars needs Θ(N) — you can't compress independent
information below its entropy. A shared vector of d < N dims recovers each item
with crosstalk `√(active/d)`. Measured: the evidence is **75% dense** (active ≈ N),
so d=2²⁰ gives coin-flip recovery, and you'd need d ≳ N (more memory than the exact
array) to recover it. Entropy of the evidence is 2.66 bits/weight; the 3-bit array
is already near that floor. Superposition adds crosstalk on top of entropy → always
worse for dense data.

LLMs superpose 10⁵ features in 4096 dims because features are **sparse per token**
(few active at once) and time-multiplexed. The evidence is the opposite: 75%
active, all the time, persistent. `capacity ∝ d/active`; here active ≈ total.

## The open lever
The only way superposition could help is to make evidence **sparse per step** —
accumulate only for the few weights with strong consistent signal, keep the rest
truly zero. Gradient noise currently keeps 75% nonzero. Suppress that (a
significance gate) and the sparse-storage / sketch idea becomes viable. Untested.
