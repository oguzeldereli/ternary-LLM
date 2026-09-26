#!/usr/bin/env bash
# Laptop part of the from-scratch 20M comparison: baseline with two more flip seeds (noise floor), stuck mask.
set -u
cd "$(dirname "$0")/../.."
for spec in "s20_base_seed1 --flip_seed 1" "s20_base_seed2 --flip_seed 2" "s20_maskstuck --lr_mask_stuck"; do
  set -- $spec; n=$1; shift
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --stop_after 611 --eval_interval 200 --save_secs 100000 "$@"
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
done
