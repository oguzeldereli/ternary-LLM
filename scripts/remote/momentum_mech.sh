#!/usr/bin/env bash
# 4090, next to whatever runs: the momentum-mechanism test on the no-look-ahead bench (nola_lab step 4000, warm
# momentum): true gradient (256 batches), then V0 plain / V1 target point / V2 transport / V3 both, 66 steps each.
# Output: ~/ternary-sync/runs/momentum_mech/momentum_mech.txt
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
D=$S/bench_nola; mkdir -p $D
[ -f $D/ckpt_4000.pt ] || cp $HOME/ternary-sync/inbox/bench_nola/ckpt_4000.pt $D/
O=$HOME/ternary-sync/runs/momentum_mech; mkdir -p $O
BENCH_CKPT=$D/ckpt_4000.pt BENCH_DIR=$D python -m scripts.analysis.momentum_mech V0,V1,V2,V3 66 2>&1 \
  | grep -v Warn > $O/momentum_mech.txt
echo "$(date '+%F %T') DONE momentum_mech"
