#!/usr/bin/env bash
# Laptop (12 GB GPU): wave test at nola_lab @4500 with momentum memory 0.97 / 0.8 / 0.5 (25% of coordinates, on CPU).
# Output: checkpoints/momentum_mech/wave4500_b{097,08,05}.txt
cd "$(dirname "$0")/../.." || exit 1
for b in 0.97 0.8 0.5; do
  WAVE_BETA=$b WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_b${b/./}.txt
done
