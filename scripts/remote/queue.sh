#!/usr/bin/env bash
# Experiment queue on the remote 4090. Runs write to ~/ternary-LLM/checkpoints (home: survives the
# booking; monitored from the laptop via knuckles). Edit this list, not the running copy.
set -u
cd "$(dirname "$0")/../.."
source /scratch0/$USER/env.sh
nvidia-smi --query-gpu=name,temperature.gpu,power.draw --format=csv,noheader
# 1. stateless cross-batch look-ahead x2 at 0.007 (the flip count the momentum run ends up with):
#    does a smaller step alone reproduce the momentum result (4.892 at 10M; look-ahead at 0.02: 5.002)?
scripts/train/screen_10m.sh r4090_la_xb2_r007 --lookahead 2 --lookahead_xbatch --rate_peak 0.007 \
  --max_temp 90 --resume_temp 85
echo "$(date '+%F %T') DONE r4090_la_xb2_r007 $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/r4090_la_xb2_r007/train.log)"
