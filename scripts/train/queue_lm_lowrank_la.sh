#!/usr/bin/env bash
# Rank-256 adaptive momentum proposes, cross-batch look-ahead x2 keeps (base: la_xb2, 5.002)
set -u
cd "$(dirname "$0")/../.."
while kill -0 856403 2>/dev/null; do sleep 10; done
scripts/train/screen_10m.sh lm_lowrank256_adapt_xb2 --lookahead 2 --lookahead_xbatch --lowrank 256 --lr_adapt --ckpt_skip 2
