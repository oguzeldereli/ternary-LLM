#!/usr/bin/env bash
# Rank-256 momentum on the LM: is the 7.3 stall from too slow or too fast a rate?
set -u
cd "$(dirname "$0")/../.."
scripts/train/screen_10m.sh lm_lowrank256_r04 --lookahead 0 --lowrank 256 --rate_peak 0.04 --ckpt_skip 2
echo ALLDONE > checkpoints/lm_lowrank_rate.done
