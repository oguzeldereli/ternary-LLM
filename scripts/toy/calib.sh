#!/usr/bin/env bash
# Calibration: does the toy form induction heads with full precision / master at all? (vocab 2048 and 256)
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  local n=$1 kind=$2 v=$3; shift 3
  local tag=""; [ "$v" != 2048 ] && tag=_v$v
  STEPS=5000 TOY_TAG=$tag scripts/toy/run.sh $n "$@"
  TOY_V=$v python -m scripts.toy.eval checkpoints/toy/$n $kind 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
}
run fp32_v2048 fp32 2048 --mode fp32 &
run fp32_v256 fp32 256 --mode fp32 &
run master_v256 master 256 --mode master --master_dtype fp32 &
wait
