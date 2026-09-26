#!/usr/bin/env bash
# Momentum fixes from scratch to 20M tokens (611 steps), momentum + cross-batch look-ahead x2 settings, same
# batches, two at a time: baseline, sign gate, spend, refresh, vnorm, low-rank magnitude add / mul.
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
run() {
  local n=$1; shift
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --lowrank 256 --ckpt_skip 2 \
    --stop_after 611 --eval_interval 200 --save_secs 100000 "$@"
  rm -f checkpoints/$n/ckpt.pt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)"
}
run s20_base & run s20_gate --lr_gate & wait
run s20_spend2 --lr_spend 2 & run s20_refresh16 --lr_refresh 16 --lr_refresh_every 10 & wait
run s20_vnorm99 --lr_vnorm 0.99 & run s20_magadd4 --lowrank_mag add:4 & wait
run s20_magmul4 --lowrank_mag mul:4
