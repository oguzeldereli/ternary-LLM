#!/usr/bin/env bash
# Rank-256 momentum with angle-scaled decay (beta_t = 0.97 * cos(g, M)), after the r=0.1 run
set -u
cd "$(dirname "$0")/../.."
while kill -0 838463 2>/dev/null; do sleep 10; done
scripts/train/screen_10m.sh lm_lowrank256_adapt --lookahead 0 --lowrank 256 --lr_adapt --ckpt_skip 2
