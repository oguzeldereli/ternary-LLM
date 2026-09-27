#!/usr/bin/env bash
# After magadd16_qk: multiplicative low-rank magnitude (rank 4) from scratch to 60M with checkpoints every 250 steps,
# so its attention heads can be measured.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
until grep -q "DONE magadd16_qk" checkpoints/remote_queue.log 2>/dev/null; do sleep 60; done
n=mul60_heads
scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 --stop_after 1831 \
  --eval_interval 250 --save_secs 3600 --lowrank_mag mul:4 --snap_every 250
echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
