#!/usr/bin/env bash
# Lab PC: the swing test for master weights (curve_master @5000, 25% of coordinates).
# Output: ~/ternary-sync/runs/momentum_mech/master_wave_5000.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
python -u -m scripts.analysis.master_wave curve_master 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/master_wave_5000.txt
