#!/usr/bin/env bash
# Lab PC: the swing test with every analysis on the runs' own checkpoints @5000 (same point as master's):
# gate + Adam step (its rule), then rc_s0 (plain). 25%. Output: ~/ternary-sync/runs/momentum_mech/own5000_*.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
cp ~/ternary-LLM/scripts/analysis/wave.py ~/ternary-LLM/scripts/analysis/testbench.py scripts/analysis/
O=$HOME/ternary-sync/runs/momentum_mech
WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave gatevnorm_rc_s0 5000 2>&1 | grep --line-buffered -v Warn > $O/own5000_gatevnorm.txt
WAVE_SUB=0.25 python -u -m scripts.analysis.wave rc_s0 5000 2>&1 | grep --line-buffered -v Warn > $O/own5000_rc_s0.txt
