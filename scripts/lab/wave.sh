#!/usr/bin/env bash
# Lab PC: slow-wave test (autocorrelation of the true gradient over 40 steps of flips) at nola_lab @164M.
# Output line by line: ~/ternary-sync/runs/momentum_mech/wave.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
[ -f checkpoints/nola_lab/ckpt_5000.pt ] || { echo "missing checkpoints/nola_lab/ckpt_5000.pt"; exit 1; }
while pgrep -u "$USER" -f "[s]cripts.analysis.(oracle_steps|valley)" >/dev/null; do sleep 20; done
python -u -m scripts.analysis.wave nola_lab 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave.txt
