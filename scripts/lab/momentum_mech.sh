#!/usr/bin/env bash
# Lab PC, next to the running job: the momentum-mechanism test on the no-look-ahead bench (nola_lab step 4000, warm
# momentum), V0 plain / V1 target point / V2 transport / V3 both, 66 steps each. Output line by line to
# ~/ternary-sync/runs/momentum_mech/momentum_mech_lab.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
D=$S/bench_nola; mkdir -p $D
[ -f $D/ckpt_4000.pt ] || cp $HOME/ternary-sync/inbox/bench_nola/ckpt_4000.pt $D/
O=$HOME/ternary-sync/runs/momentum_mech; mkdir -p $O
BENCH_CKPT=$D/ckpt_4000.pt BENCH_DIR=$D python -u -m scripts.analysis.momentum_mech V0,V1,V2,V3 66 2>&1 \
  | grep --line-buffered -v Warn > $O/momentum_mech_lab.txt
