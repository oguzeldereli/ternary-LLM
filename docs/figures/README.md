# Figures

Grouped by the question each one answers. Superseded figures from the stateless-flip phase live in
`archive/` (their scripts still write there).

## Where we stand
| figure | shows | script |
|---|---|---|
| `live.png` | current long runs vs master and fp32: train loss, val ppl, local slope, token stretch | `plot_live.py` |
| `all_runs.png` | every long run on one set of axes | `plot_all.py` |
| `summary.png` / `summary.pdf` | one-page progress figure (sent to the professor) | `plot_summary.py` |

## Which fixes help (20M from-scratch screens)
| figure | shows | script |
|---|---|---|
| `s20.png` | baseline seeds, gate, vnorm, spend, refresh, additive/multiplicative magnitude, stuck mask at 20M | `plot_s20.py` |

## Why momentum lags master (decision quality)
| figure | shows | script |
|---|---|---|
| `m_diag.png` | per-step |g|, |M|, sign agreement, subspace share over training | `plot_m_diag.py` |
| `m_tracking.png` | how well M tracks the gradient as training goes | `plot_m_tracking.py` |
| `windows.png`, `window_curves.png` | agreement of 1..100-step net moves with the summed gradient, ours vs master | `plot_windows.py`, `plot_window_curves.py` |
| `momentum_off.png` | switching momentum off at 11M falls back to the look-ahead-only curve | `plot_momentum_off.py` |
| `branch340.png`, `overnight_branches.png` | branch tests: M vs count-matched g proposals, beta 0.99/0.995 | `plot_branch340.py`, `plot_overnight_branches.py` |

## Induction heads
| figure | shows | script |
|---|---|---|
| `toy_induction.png` | 2-layer toy on repeated random segments: induction (shuffled - exact loss) and val for every optimizer arm | `plot_toy.py` |

## Other
| figure | shows | script |
|---|---|---|
| `mlp.png` | MLP width / N+K scaling sketch | — |
