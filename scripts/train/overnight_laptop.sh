#!/usr/bin/env bash
# Laptop overnight (after s20_base_seed1): fix trials first, the seed last, then a longer gate run.
set -u
cd "$(dirname "$0")/../.."
while kill -0 1705984 2>/dev/null; do sleep 20; done
rm -f checkpoints/s20_base_seed1/ckpt.pt
echo "$(date '+%F %T') DONE s20_base_seed1 $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/s20_base_seed1/train.log)"
run() {
  local n=$1 steps=$2; shift 2
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --stop_after $steps --eval_interval 200 --save_secs 3600 "$@"
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run s20_maskstuck 611 --lr_mask_stuck
run s20_gate_maskstuck 611 --lr_gate --lr_mask_stuck
run s20_gate_vnorm99 611 --lr_gate --lr_vnorm 0.99
run s20_base_seed2 611 --flip_seed 2
run s60_gate 1831 --lr_gate
