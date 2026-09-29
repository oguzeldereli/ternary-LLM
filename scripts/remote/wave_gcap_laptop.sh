#!/usr/bin/env bash
# Laptop: the user's rule with the cap in gradient units (||M|| <= C x mean batch-gradient norm), no friction,
# absolute tanh flip chance, C = 3 and 1, nola_lab @4500, 25% of coordinates.
cd "$(dirname "$0")/../.." || exit 1
for c in 3 1; do
  WAVE_BETA=1 WAVE_GCAP=$c WAVE_PFUN=tanh WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_user_gcap${c}_laptop.txt
done
