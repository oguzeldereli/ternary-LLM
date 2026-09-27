#!/usr/bin/env bash
# Induction toy (vocab 256 data, 15000 steps): master control, then rank-64 momentum with / without additive
# magnitude and look-ahead, all in parallel.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  local n=$1 kind=$2; shift 2
  STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n "$@"
  TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n $kind 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
}
run master master --mode master --master_dtype fp32 &
run mom kernel --lowrank 64 &
run mom_add kernel --lowrank 64 --lowrank_mag add:4 &
run mom_la kernel --lowrank 64 --lookahead 2 --lookahead_xbatch &
run mom_add_la kernel --lowrank 64 --lookahead 2 --lookahead_xbatch --lowrank_mag add:4 &
wait
