#!/usr/bin/env bash
# Lab PC: where momentum loses alignment once flips start (nola_lab @164M). Output line by line:
# ~/ternary-sync/runs/momentum_mech/mech_why.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
mkdir -p checkpoints/nola_lab
[ -f checkpoints/nola_lab/ckpt_5000.pt ] || cp $HOME/ternary-sync/inbox/nola5000/ckpt_5000.pt checkpoints/nola_lab/
python -u -m scripts.analysis.mech_why nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/mech_why.txt
