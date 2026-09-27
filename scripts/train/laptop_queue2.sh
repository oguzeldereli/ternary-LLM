#!/usr/bin/env bash
# Laptop queue after bend_fp32_w900: per-head attention temperature test (from scratch to 60M, checkpoints every
# 250 steps, heads/induction measured on each), then ours with probes to 30M, then the remaining fix trials.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 2109508 2>/dev/null; do sleep 20; done
rm -f checkpoints/bend_fp32_w900/ckpt.pt
echo "$(date '+%F %T') DONE bend_fp32_w900 $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/bend_fp32_w900/train.log)"
run() {
  local n=$1 steps=$2; shift 2
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --stop_after $steps --eval_interval 250 --save_secs 3600 "$@"
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run qktemp60 1831 --qk_temp --snap_every 250
python -m scripts.analysis.heads_over_time checkpoints/qktemp60 > checkpoints/qktemp60/heads.txt 2>&1
rm -f checkpoints/qktemp60/ckpt.pt
run bend_ours 916 --probe 0-920:25; rm -f checkpoints/bend_ours/ckpt.pt
run s20_gate_vnorm99 611 --lr_gate --lr_vnorm 0.99; rm -f checkpoints/s20_gate_vnorm99/ckpt.pt
run s20_base_seed2 611 --flip_seed 2; rm -f checkpoints/s20_base_seed2/ckpt.pt
run s60_gate 1831 --lr_gate
