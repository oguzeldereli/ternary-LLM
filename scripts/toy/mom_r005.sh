#!/usr/bin/env bash
# Induction toy: momentum only (no look-ahead) with the peak flip rate cut 4x (0.02 -> 0.005).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
n=mom_r005
STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n --lowrank 64 --rate_peak 0.005
TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
