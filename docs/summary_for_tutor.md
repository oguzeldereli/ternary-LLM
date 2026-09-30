# Training ternary LLMs without full-precision weights

*One-page summary, 30 Sep 2026*

## The question

Can a language model whose weights are ternary (each weight is -1, 0 or +1) be trained without a full-precision copy
of every weight? Standard ternary training (BitNet-style) keeps that copy, the "master weights", plus two Adam states
per weight: about 16 bytes per weight during training, roughly 430 GB for a 27B model.

My aim is an optimizer whose memory grows slower than the number of weights, where the weights change only by discrete
flips. The long-term target is training a 27B ternary model on a 12-16 GB consumer GPU.

## What I built

- **Flip rule.** Each step, a weight moves one step (-1 -> 0 -> +1) with a probability proportional to the size of its
  update signal, in the direction that lowers the loss. There is no latent weight; the ternary values are the only
  per-weight state.
- **Low-rank momentum.** The update signal is a momentum of the gradient, stored per layer as a rank-r factorization
  (r x (rows + columns) numbers, not rows x columns), kept up to date with one subspace-iteration step per batch.
- **Sign gate.** A weight flips only where the momentum and the current batch gradient agree in sign.
- **Factored Adam step.** The momentum is divided by a row x column estimate of the gradient's second moment (as in
  Adafactor), which moves flips out of steep rows.
- **Long memory.** The key fix from this week: the momentum must remember ~200-400 steps, not ~33. Either a decay of
  0.995, or "dry friction": the momentum's total size is reduced by a fixed fraction of the gradient's size each step,
  which acts as a decay that sets its own memory length.
- **Spend.** When a weight flips, the push that caused it is removed from the momentum.

The optimizer state is O(r x (rows + columns)) per layer: sublinear in the parameter count.

## Results so far

110M-parameter model, 300M tokens of Wikipedia (32k vocabulary), everything trained from scratch, one seed:

| method | validation loss | perplexity |
|---|---|---|
| full precision (fp32) | 2.683 | 14.6 |
| ternary with master weights (the standard method) | 2.751 | 15.7 |
| **mine: long memory (dry friction) + spend + rank-512 momentum** | **2.822** | **16.8** |
| mine, decay 0.995 + rank 512 | 2.834 | 17.0 |
| mine, before the long-memory fix | 2.988 | 19.8 |
| mine, two days ago | 3.077 | 21.7 |
| mine, plain low-rank momentum | 3.131 | 22.9 |

The gap to master weights went from 0.33 to 0.07 in loss (perplexity 21.7 -> 16.8, master 15.7).

## What I found

- **Where the gap was.** Splitting the validation loss by how often each (previous word, next word) pair appears in
  training showed the gap sat in pairs seen fewer than ~10,000 times; frequent pairs were learned as well as master.
- **Why.** A pair seen 100 times in training turns up about once every 120 steps. A momentum with a ~33-step memory
  forgets each sighting before the next one comes, so rare signals never accumulate. Master weights keep a running sum
  per weight, so they do. Lengthening the memory to ~200-400 steps halved the rare-pair gap.
- **It is a memory problem, not a rank problem.** The rank-256 momentum already held 78-89% of the rare-pair gradient;
  what was missing was how long it was remembered. Rank 512 then adds ~0.05; rank 1024 adds little more.
- **Induction heads.** With the long memory, the models form strong induction (copying) heads, which the earlier
  versions did not.
- **Memory length has an optimum.** Decay 0.97 / 0.99 / 0.995 / 0.998 / 1.0 gives 2.99 / 2.91 / 2.88 / 2.89 / 3.05;
  too long keeps stale early gradients.

## What is not done yet

- One seed only; a second seed is needed to confirm the 0.07 gap.
- Scale: a 340M model (recipe and master) is training now.
- The rank needed at large widths: at 27B, a 12 GB card leaves room for about rank 64, a 16 GB card about rank 256. A
  rank sweep (32 / 64 / 128 at 110M and 340M) is running to see whether the rank needed grows with model width.
- The last 0.07 of the gap to master.
- The trainer currently holds every layer's full gradient at once; for 27B it must update each layer during the
  backward pass.

## Plan and what I would like to ask

- 8-10 Oct: a 3-day booking of a 2 x RTX PRO 6000 workstation (goldbug): my method against master weights at ~1B
  parameters on ~2B tokens, plus a 27B memory and speed test. I also have access to Myriad (A100s).
- I would value your advice on whether this is worth writing up, and who in the department works closest to it
  (optimisation, efficient training of LLMs), for feedback and possibly supervision.
- Later: help with an arXiv endorsement for a preprint after the goldbug results.

Code and all run logs: the ternary-LLM repository (every run is dated in its history).
