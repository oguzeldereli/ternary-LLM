#!/usr/bin/env bash
# Laptop, overnight: the swing test on trained checkpoints (25% of coordinates), same step, each with its own rule:
#   rc_s0 @5000 (old rule) vs speedref_rc_s0 @5000 (--speed_ref 0.995 rule), then @8000 for both.
# Checkpoints are pulled from the 4090's scratch (not through home). Output: checkpoints/momentum_mech/probe_*.txt
cd "$(dirname "$0")/../.." || exit 1
J="-o BatchMode=yes -J oguzelde@knuckles.cs.ucl.ac.uk"; H=oguzelde@beachcomber.cs.ucl.ac.uk:/scratch0/oguzelde/runs
pull() {  # RUN STEP: wait until the 4090 has it, then copy
  mkdir -p checkpoints/$1
  until [ -f checkpoints/$1/ckpt_$2.pt ]; do
    scp -q $J $H/$1/ckpt_$2.pt checkpoints/$1/ 2>/dev/null || { rm -f checkpoints/$1/ckpt_$2.pt; sleep 300; }
  done
}
probe() { # RUN STEP NORM
  pull $1 $2
  WAVE_NORM=$3 WAVE_SUB=0.25 python -u -m scripts.analysis.wave $1 $2 2>&1 | grep --line-buffered -v Warn \
    > checkpoints/momentum_mech/probe_$1_$2_laptop.txt
}
probe rc_s0 5000 own
probe speedref_rc_s0 5000 ema:0.995
probe rc_s0 8000 own
probe speedref_rc_s0 8000 ema:0.995
