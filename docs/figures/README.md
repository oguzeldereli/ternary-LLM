# Figures

All current figures are for the 110M model (`small`), 300M tokens of wiki32k, from scratch, no look-ahead unless
noted. Validation loss is in nats per token; perplexity = e^loss. Formulas behind each rule: [../FORMULAS.md](../FORMULAS.md).

## Current

| figure | shows | script |
|---|---|---|
| `best_runs.png` | the best run at each step of progress, with the fp32 and master-weights references; whole run (log tokens) and 100M-300M with final loss and perplexity | `python -m scripts.plots.plot_best` |
| `scale.png` | the recipe at 110M and 340M: the 110M rank sweep (32-1024, int8, refresh), 340M runs against 110M with master at both sizes, final loss against rank | `python -m scripts.plots.plot_scale` |
| `sweeps.png` | the recipe's three knobs, final loss: momentum decay (as memory 1/(1-b)), dry friction strength, momentum rank | `python -m scripts.plots.plot_sweeps` |
| `gaps.png` | where the gap to master is at 300M: by training-set count of the (previous, target) pair, by context position (small buckets are noisy: 96 and 672 tokens), and the copy (induction) gain | `python -m scripts.plots.plot_gaps` (data: `scripts/analysis/loss_by_freq.py`, `loss_by_pos.py`) |
| `night_runs.png` | every run of the night of 29-30 Sep on the sharp base, final loss as the gap to master | `python -m scripts.plots.plot_night` |
| `swing_tests.png` | 41-step swing tests from one checkpoint: loss change vs flips, the lag-3 reversal of the true gradient, uphill share, never-turning weights, against master | `python -m scripts.plots.plot_swing` (data: `scripts/analysis/wave.py`, `master_wave.py`) |
| `mlp.png` | MLP width / N+K scaling sketch (the user's) | `scripts/plots/plot_mlp.py` |

## Archive (`archive/`): cited in RUNS.md, not updated

| figure | shows |
|---|---|
| `categories/*.png` | one figure per category tried up to 29 Sep (look-ahead, magnitude, attention, momentum fixes, step size, mechanisms, speed and gate, the user's rule); references were the look-ahead baseline and master (`plot_categories.py`, now writes here) |
| `focus_runs.png` | the live view of 29-30 Sep (`plot_focus.py`, now writes here) |
| `induction_text.png`, `toy_induction.png` | induction probes on text runs and on the 2-layer toy (27-28 Sep) |
| `bench_mechanisms.png`, `toy_mechanisms.png` | the mechanism bench (28 Sep) |
| `all_runs.png` | every run up to 27 Sep, grouped, including the stateless-flip phase |
| `summary.png` / `.pdf` | the one-page progress figure of 26 Sep |
| `m_diag.png`, `m_tracking.png` | momentum vs gradient per step (\|g\|, \|M\|, agreement, subspace share) |
| `windows.png`, `window_curves.png` | agreement of 1..100-step net moves with the summed gradient, ours vs master |
| `momentum_off.png` | switching momentum off at 11M falls back to the look-ahead-only curve |
| `branch340.png`, `overnight_branches.png` | branch tests: M vs count-matched g proposals, beta 0.99 / 0.995 |

Deleted on 30 Sep (one-off views, superseded by `best_runs.png` and `night_runs.png`): `today_runs.png`,
`sharp_run.png`, `undo_run.png` (their scripts are in `scripts/plots/legacy/`). Deleted on 27 Sep: the stateless-flip
phase figures, `live.png`, `s20.png`.
