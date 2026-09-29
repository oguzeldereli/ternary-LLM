#!/usr/bin/env bash
# 4090: the user's rule in the swing test at nola_lab @4500 (25% of coordinates): momentum = pure sum of gradients
# (beta 1, no decay), per-weight cap at CAP x its starting mean size, flip chance on a fixed (absolute) scale.
cd "$HOME/ternary-LLM" || exit 1
source /scratch0/$USER/env.sh
for c in 3 1; do
  WAVE_BETA=1 WAVE_CAP=$c WAVE_NORM=fixed WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_userrule_cap${c}_4090.txt
done
