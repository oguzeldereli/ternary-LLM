#!/usr/bin/env bash
# Induction toy: momentum + additive + look-ahead with the adapter's noise walk held back three ways.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  local n=$1; shift
  STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n --lowrank 64 --lookahead 2 --lookahead_xbatch --lowrank_mag add:4 "$@"
  MAG_KIND=add TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
}
run add_la_wd01 --mag_wd 0.1 &
run add_la_lr01 --mag_lr_mult 0.1 &
run add_la_cap05 --mag_cap 0.5 &
wait
