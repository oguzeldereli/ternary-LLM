#!/usr/bin/env bash
# Induction toy replicates (seeds 2 and 3: init, data order, look-ahead batches) of the three arms the adapter
# conclusion rests on, plus one stacking arm: low flip rate (0.005) with look-ahead and the weight-decayed adapter.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  local n=$1; shift
  STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n "$@"
  MAG_KIND=add TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
}
LA="--lowrank 64 --lookahead 2 --lookahead_xbatch"
for s in 2 3; do
  run add_la_wd01_s$s $LA --lowrank_mag add:4 --mag_wd 0.1 --seed $s &
  run mom_add_la_s$s $LA --lowrank_mag add:4 --seed $s &
  run mom_la_s$s $LA --seed $s &
done
run add_la_wd01_r005 $LA --lowrank_mag add:4 --mag_wd 0.1 --rate_peak 0.005 &
wait
