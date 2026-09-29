#!/usr/bin/env bash
# Lab PC: does each rule damp the swing, and is the overshoot within one step? 25% of coordinates.
# Output: ~/ternary-sync/runs/momentum_mech/damp_*.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=$HOME/ternary-sync/runs/momentum_mech
WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/damp_plain_nola4500.txt
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/damp_gatevnorm_nola4500.txt
WAVE_SIG=gate WAVE_SUB=0.25 python -u -m scripts.analysis.wave gate_rc_s0 8000 2>&1 | grep --line-buffered -v Warn > $O/damp_gate_rc_s0_8000.txt
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave gatevnorm_rc_s0 8000 2>&1 | grep --line-buffered -v Warn > $O/damp_gatevnorm_rc_s0_8000.txt
