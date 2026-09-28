#!/usr/bin/env bash
# Lab PC: wave test controls at nola_lab @164M: frozen weights (rate 0: is the oscillation made by the flips?) and
# 1/4 rate (does it scale with the step?). Output: ~/ternary-sync/runs/momentum_mech/wave_r0.txt, wave_r025.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
while pgrep -u "$USER" -f "[s]cripts.analysis.(oracle_steps|wave|flip_choice)" >/dev/null; do sleep 20; done
for m in 0 0.25; do
  WAVE_RATE_MULT=$m python -u -m scripts.analysis.wave nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave_r${m/./}.txt
done
