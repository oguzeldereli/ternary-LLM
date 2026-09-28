#!/usr/bin/env bash
# Lab PC: which weights to flip along the true gradient (size / consistency / adam / adam factored / random), at
# nola_lab @164M, follow-up. Output: ~/ternary-sync/runs/momentum_mech/flip_choice2.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
while pgrep -u "$USER" -f "[s]cripts.analysis.(oracle_steps|wave)" >/dev/null; do sleep 20; done
python -u -m scripts.analysis.flip_choice2 nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/flip_choice2.txt
