#!/usr/bin/env bash
# 4090: the user's rule with friction back (memory ~30 steps) and the undo, nola_lab @4500, 25% of coordinates.
#   1 beta 0.97 + norm cap 1x start + absolute tanh     2 + undo     3 beta 0.9 + cap + tanh + undo
#   4 plain rule + undo only                             5 beta 0.97 + cap + tanh + undo + gate + Adam step
cd "$HOME/ternary-LLM" || exit 1
source /scratch0/$USER/env.sh
O=checkpoints/momentum_mech
w() { local out=$1; shift; env "$@" WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/$out; }
w wave4500_fu1_b097_cap_tanh_4090.txt WAVE_NCAP=1 WAVE_PFUN=tanh
w wave4500_fu2_b097_cap_tanh_undo_4090.txt WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_UNDO=1
w wave4500_fu3_b09_cap_tanh_undo_4090.txt WAVE_BETA=0.9 WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_UNDO=1
w wave4500_fu4_plain_undo_4090.txt WAVE_UNDO=1
w wave4500_fu5_all_4090.txt WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_UNDO=1 WAVE_SIG=gate+vnorm:0.99
