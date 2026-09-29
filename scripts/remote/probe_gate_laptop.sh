#!/usr/bin/env bash
# Laptop: swing test with per-weight analysis on the gate run's own checkpoint (gate rule) and on rc_s0 (old rule),
# both @5000, 25% of coordinates. Output: checkpoints/momentum_mech/probe2_*_laptop.txt
cd "$(dirname "$0")/../.." || exit 1
WAVE_SIG=gate WAVE_SUB=0.25 python -u -m scripts.analysis.wave gate_rc_s0 5000 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/probe2_gate_rc_s0_5000_laptop.txt
WAVE_SUB=0.25 python -u -m scripts.analysis.wave rc_s0 5000 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/probe2_rc_s0_5000_laptop.txt
