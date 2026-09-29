#!/usr/bin/env bash
# Laptop: after probe_gate_laptop.sh, the swing test with per-weight analysis on gatevnorm_rc_s0 @5000 (its own rule).
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "[p]robe_gate_laptop.sh" >/dev/null; do sleep 30; done
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave gatevnorm_rc_s0 5000 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/probe2_gatevnorm_rc_s0_5000_laptop.txt
