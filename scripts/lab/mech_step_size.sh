#!/usr/bin/env bash
# Lab PC: overshoot line search + flip-rate sweep at nola_lab @164M. Output: ~/ternary-sync/runs/momentum_mech/step_size.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
python -u -m scripts.analysis.mech_step_size nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/step_size.txt
