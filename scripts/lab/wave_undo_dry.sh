#!/usr/bin/env bash
# Lab PC: swing tests at nola_lab @4500 (25%): the old rule + undo (moved from the 4090), then dry friction (a fixed
# amount off per step instead of x beta; 1/33: one gradient gone after 33 steps), beta 1, norm cap 1x, absolute tanh:
# on the whole vector, then per weight. Output: ~/ternary-sync/runs/momentum_mech/wave4500_*_lab.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
O=$HOME/ternary-sync/runs/momentum_mech
w() { local out=$1; shift; env "$@" WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/$out; }
w wave4500_fu4_plain_undo_lab.txt WAVE_UNDO=1
w wave4500_dry_vec_lab.txt WAVE_BETA=1 WAVE_DRY=0.0303 WAVE_NCAP=1 WAVE_PFUN=tanh
w wave4500_dry_w_lab.txt WAVE_BETA=1 WAVE_DRYW=0.0303 WAVE_NCAP=1 WAVE_PFUN=tanh
