#!/usr/bin/env bash
# Laptop, after the gradient-unit cap tests: does each rule damp the swing properly?
#   1. gate + Adam step from the same branch point as every other swing test (nola_lab @4500)
#   2-3. gate and gate + Adam step on their own checkpoints late in training (@8000)
# 25% of coordinates. Output: checkpoints/momentum_mech/damp_*.txt
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "[w]ave_gcap_laptop.sh" >/dev/null; do sleep 30; done
O=checkpoints/momentum_mech
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/damp_gatevnorm_nola4500.txt
WAVE_SIG=gate WAVE_SUB=0.25 python -u -m scripts.analysis.wave gate_rc_s0 8000 2>&1 | grep --line-buffered -v Warn > $O/damp_gate_rc_s0_8000.txt
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave gatevnorm_rc_s0 8000 2>&1 | grep --line-buffered -v Warn > $O/damp_gatevnorm_rc_s0_8000.txt
