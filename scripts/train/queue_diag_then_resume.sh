#!/usr/bin/env bash
# After the step-0..345 diagnostic reproduction: 8-batch snapshot at step 340 (~10^7.05 tokens),
# then resume the main 300M run with per-step diagnostics.
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "out_dir checkpoints/lm_lowrank256_xb2_diag " >/dev/null; do sleep 10; done
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.analysis.m_snapshot \
  checkpoints/lm_lowrank256_xb2_diag/ckpt_340.pt 8 > checkpoints/lm_lowrank256_xb2_diag/snapshot_340.txt 2>&1
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m scripts.analysis.m_snapshot \
  checkpoints/lm_lowrank256_xb2_100M/ckpt_6243.pt 8 > checkpoints/lm_lowrank256_xb2_100M/snapshot_6243.txt 2>&1
exec scripts/train/continue_lowrank_xb2_300M.sh
