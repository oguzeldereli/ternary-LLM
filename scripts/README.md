# scripts

Run everything from the repo root. Python scripts are modules (`python -m scripts.<dir>.<name>`);
shell launchers can be called from anywhere.

| dir | contents |
|---|---|
| `train/` | `screen_10m.sh` (10M-token screen of the best config), `full_run.sh` (300M tokens); `legacy/` = the queues used for earlier studies (`orchestrate.sh`, `ef_overnight.sh`, `lockout_overnight.sh`, `run_sweep_3b.sh`) |
| `analysis/` | offline experiments on checkpoints: `lookahead_test` (one-step look-ahead filter), `efficiency_test` (per-step gain: signal vs curvature), `curv_single` (per-weight curvature), `curv_validate`, `hessian_probe`, `diag`, `diag_snr` (gradient SNR), `superpose_*` |
| `plots/` | figures into `docs/figures/`: `plot_all` (every run grouped), `plot_live` (long look-ahead run), `plot_early` (first 33M tokens), `plot_probes`, `plot_stretch`, `plot_tail`; `plot_run`, `plot_compare` are per-run tools; `legacy/` = retired figures |
| `bench/` | kernel and throughput benchmarks (`bench_b27` = 27B) |
| `data/` | `prep_wiki` (tokenize Wikipedia) |
| `generate/` | sample text from checkpoints |
