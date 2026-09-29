#!/usr/bin/env bash
# Lab PC: swing test (all coordinates) at nola_lab @4500, beta 0.97, flip-signal arms that act per direction:
# sign gate (reacts to a turn at once), factored-Adam step (smaller steps where steep), per-row speed reference.
# Output: ~/ternary-sync/runs/momentum_mech/wave4500_full_<arm>.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
for a in gate vnorm:0.99 rowema:0.995; do
  WAVE_SIG=$a python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave4500_full_${a/:/}.txt
done
