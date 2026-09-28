#!/usr/bin/env bash
# 4090: wave test at nola_lab @4500, beta 0.97, 25% of coordinates: the trainer's flip rule (momentum divided by its
# own current size) vs a fixed divisor (its size at step 0: fewer flips when the momentum shrinks, like a real mass).
# Output: checkpoints/momentum_mech/wave4500_norm_{own,fixed}.txt
mkdir -p checkpoints/momentum_mech
for n in own fixed; do
  WAVE_NORM=$n WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_norm_$n.txt
done
