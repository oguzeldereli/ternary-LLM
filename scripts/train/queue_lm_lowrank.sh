#!/usr/bin/env bash
# LM 10M screens: plain flips vs rank-256 momentum vs frozen ternary (rate 0), same batches
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "scripts.mlp.geometry" >/dev/null; do sleep 10; done
scripts/train/screen_10m.sh lm_lowrank256 --lookahead 0 --lowrank 256 --ckpt_skip 2
scripts/train/screen_10m.sh lm_plain --lookahead 0 --ckpt_skip 2
scripts/train/screen_10m.sh lm_frozen --lookahead 0 --rate_peak 0 --ckpt_skip 2
echo ALLDONE > checkpoints/lm_lowrank_queue.done
