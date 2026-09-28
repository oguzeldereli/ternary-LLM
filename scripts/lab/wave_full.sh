#!/usr/bin/env bash
# Lab PC: full swing test (all coordinates) at nola_lab @4500, beta 0.97: the old flip rule (momentum divided by its
# current size), a fixed divisor (its size at step 0), the trainer's --speed_ref 0.995 (slow EMA).
# Output: ~/ternary-sync/runs/momentum_mech/wave4500_full_{own,fixed,ema0.995}.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
[ -f checkpoints/nola_lab/ckpt_4500.pt ] || { echo "missing ckpt_4500"; exit 1; }
for n in own fixed ema:0.995; do
  WAVE_NORM=$n python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave4500_full_${n/:/}.txt
done
