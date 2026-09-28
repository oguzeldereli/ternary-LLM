#!/usr/bin/env bash
# Lab PC: does a shorter momentum memory damp the flip-made swing? Wave test at full rate with beta 0.8 and 0.5.
# Output: ~/ternary-sync/runs/momentum_mech/wave_b08.txt, wave_b05.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
while pgrep -u "$USER" -f "[s]cripts.analysis.(oracle_steps|wave|flip_choice)" >/dev/null; do sleep 20; done
for b in 0.8 0.5; do
  WAVE_BETA=$b python -u -m scripts.analysis.wave nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave_b${b/./}.txt
done
