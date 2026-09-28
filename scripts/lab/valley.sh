#!/usr/bin/env bash
# Lab PC: the valley / zigzag test and the online-selector label test at nola_lab @164M.
# Output line by line: ~/ternary-sync/runs/momentum_mech/valley.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
mkdir -p checkpoints/nola_lab
[ -f checkpoints/nola_lab/ckpt_5000.pt ] || cp $HOME/ternary-sync/inbox/nola5000/ckpt_5000.pt checkpoints/nola_lab/
python -u -m scripts.analysis.valley nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/valley.txt
rm -f $HOME/ternary-sync/inbox/nola5000/ckpt_5000.pt
