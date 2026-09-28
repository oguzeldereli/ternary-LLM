#!/usr/bin/env bash
# Induction toy (vocab 256, 15000 steps, no look-ahead): the momentum mechanisms vs plain momentum (toy "mom", 1.07
# nats at rate 0.02 and "mom_r005", 2.48 at rate 0.005). V1 and the user's design (gain 1 and gain 0), rate 0.02.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
run() {
  local n=$1; shift
  STEPS=15000 TOY_TAG=_v256 scripts/toy/run.sh $n --lowrank 64 "$@"
  TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)"
}
run mom_mech_v1 --mech v1 &
run mom_mech_user --mech user --mech_gain 1 &
run mom_mech_user_g0 --mech user --mech_gain 0 &
wait
