#!/usr/bin/env bash
# Lab PC: overshoot line search + flip-rate sweep at nola_lab @164M. Output: ~/ternary-sync/runs/momentum_mech/step_size.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1   # never run from the home folder (10 GB quota)
python -u -m scripts.analysis.mech_step_size nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/step_size.txt
