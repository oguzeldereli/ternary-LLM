#!/usr/bin/env bash
# Laptop: alignment/noise of the trainer's mechanisms along their own runs (164M, 229M, 295M).
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for x in "U1 mech_user_b131" "U0 mech_user_g0_b131" "V1 mech_v1_b131"; do
  set -- $x; python -m scripts.analysis.mech_along $1 $2 5000,7000,9000 2>&1 | grep --line-buffered -v Warn
done > checkpoints/mech_along_laptop.txt
