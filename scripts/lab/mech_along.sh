#!/usr/bin/env bash
# Lab PC (shoveler, where nola_lab's snapshots are): plain momentum, V2 (gain 1) and V3 (bench version) along the
# no-look-ahead run (164M, 229M, 295M). Output: ~/ternary-sync/runs/momentum_mech/mech_along_lab.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
for A in V0 V2 V3; do
  python -u -m scripts.analysis.mech_along $A nola_lab 5000,7000,9000 2>&1 | grep --line-buffered -v Warn
done > $HOME/ternary-sync/runs/momentum_mech/mech_along_lab.txt
