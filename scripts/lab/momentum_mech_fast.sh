#!/usr/bin/env bash
# Fast version of the momentum-mechanism test: 128-batch truth at the bench, 33 steps per arm, 32-batch truths at
# 10 and 33 steps. Output line by line to ~/ternary-sync/runs/momentum_mech/momentum_mech_fast.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
D=$S/bench_nola_fast; mkdir -p $D
[ -f $D/ckpt_4000.pt ] || cp $HOME/ternary-sync/inbox/bench_nola/ckpt_4000.pt $D/
O=$HOME/ternary-sync/runs/momentum_mech; mkdir -p $O
BENCH_CKPT=$D/ckpt_4000.pt BENCH_DIR=$D BENCH_TRUTH=128 EVAL_TRUTH=32 python -u -m scripts.analysis.momentum_mech \
  V0,V1,V2,V3 33 2>&1 | grep --line-buffered -v Warn > $O/momentum_mech_fast.txt
