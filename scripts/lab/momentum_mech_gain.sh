#!/usr/bin/env bash
# Corrected V2 (move correction alone) and V3 (target + move correction) at gains 1 and 3 (gain 33 diverged), on the
# no-look-ahead bench (true gradient cached from the full test), 66 steps. Runs next to the training job.
# Output line by line: ~/ternary-sync/runs/momentum_mech/momentum_mech_gain.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
D=$S/bench_nola; O=$HOME/ternary-sync/runs/momentum_mech; mkdir -p $O
for G in 1 3; do
  echo "== gain $G"
  MECH_GAIN=$G BENCH_CKPT=$D/ckpt_4000.pt BENCH_DIR=$D python -u -m scripts.analysis.momentum_mech V2,V3 66 2>&1 | grep --line-buffered -v Warn
done > $O/momentum_mech_gain.txt
