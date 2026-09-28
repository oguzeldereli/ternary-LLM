#!/usr/bin/env bash
# Lab PC: how much a perfect direction is worth at the right amount (nola_lab @164M). Output:
# ~/ternary-sync/runs/momentum_mech/oracle_steps.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
python -u -m scripts.analysis.oracle_steps nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/oracle_steps.txt
