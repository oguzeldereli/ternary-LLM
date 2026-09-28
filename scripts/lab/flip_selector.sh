#!/usr/bin/env bash
# Lab PC: learned flip selectors (A aligned, B loss change) at nola_lab @164M. Waits for the analysis jobs here.
# Output line by line: ~/ternary-sync/runs/momentum_mech/flip_selector.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
while pgrep -u "$USER" -f "scripts.analysis.mech_step_size|scripts.analysis.mech_why" >/dev/null; do sleep 20; done
python -u -m scripts.analysis.flip_selector nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/flip_selector.txt
