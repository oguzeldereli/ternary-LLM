# Figures

## Current: one figure per category of what was tried (`categories/`)

Every figure has the same two dashed references, **master weights** (orange) and **our baseline, rank-256
momentum + cross-batch look-ahead x2** (black), then only that category's runs. Panels: train loss (log-log),
train loss minus the baseline's at the same tokens (below 0 = better), validation loss. Redrawn every 10 min
while runs are going (`scripts/plots/refresh_ext.sh`); by hand: `python -m scripts.plots.plot_categories`.

| figure | what was tried |
|---|---|
| `categories/look_ahead.png` | momentum with vs without look-ahead; look-ahead without momentum |
| `categories/look_ahead_schedules.png` | look-ahead switched on/off during training (off to 131M then on; on to 30M, off, on again at 98M / 131M) |
| `categories/magnitude.png` | float low-rank magnitude beside the trits: additive r4 / r16, adapter weight decay, without look-ahead, multiplicative |
| `categories/attention.png` | per-head attention temperature, head protection |
| `categories/momentum_fixes_20M.png` | 20M from-scratch screens: sign gate, vnorm, spend, refresh, stuck mask, magnitude, baseline seeds |

## Induction heads

| figure | shows | script |
|---|---|---|
| `induction_text.png` | text runs: validation loss, ordered copying (exact - shuffled repeat gain on random tokens), seen-token boost | `plot_induction_text.py` (data: `scripts/analysis/induction_track.py`) |
| `toy_induction.png` | 2-layer toy on repeated random segments, every optimizer arm | `plot_toy.py` |

## Archive (`archive/`): cited in RUNS.md, not updated

| figure | shows |
|---|---|
| `all_runs.png` | every run up to 27 Sep, grouped, including the stateless-flip phase |
| `summary.png` / `.pdf` | the one-page progress figure sent to the professor (26 Sep) |
| `m_diag.png`, `m_tracking.png` | momentum vs gradient per step (|g|, |M|, agreement, subspace share) |
| `windows.png`, `window_curves.png` | agreement of 1..100-step net moves with the summed gradient, ours vs master |
| `momentum_off.png` | switching momentum off at 11M falls back to the look-ahead-only curve |
| `branch340.png`, `overnight_branches.png` | branch tests: M vs count-matched g proposals, beta 0.99 / 0.995 |

Deleted on 27 Sep (regenerable from their scripts, no longer informative): the stateless-flip-phase figures,
`live.png` and `s20.png` (replaced by the category figures).

## Other

`mlp.png`: MLP width / N+K scaling sketch.
