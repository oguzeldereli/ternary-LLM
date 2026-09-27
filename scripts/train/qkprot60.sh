#!/usr/bin/env bash
# Per-head temperature + protection of sharpened heads (alpha 2), from scratch to 60M, checkpoints every 250
# steps, heads measured on each (compare qktemp60: head forms at 25M then flickers).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n=qkprot60
scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --stop_after 1831 \
  --eval_interval 250 --save_secs 3600 --qk_temp --qk_protect 2 --snap_every 250
python -m scripts.analysis.heads_over_time checkpoints/$n > checkpoints/$n/heads.txt 2>&1
rm -f checkpoints/$n/ckpt.pt
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
