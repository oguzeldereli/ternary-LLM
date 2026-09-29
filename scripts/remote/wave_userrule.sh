#!/usr/bin/env bash
# 4090: the user's rule in the swing test at nola_lab @4500 (25% of coordinates). Momentum = one velocity vector:
# gravity = the gradient added every step (beta 1: no friction), capped as a whole vector (||M|| <= NCAP x start),
# flip chance in absolute units rate * tanh(|M_ij| / v0) with v0 fixed (no division by the current size).
cd "$HOME/ternary-LLM" || exit 1
source /scratch0/$USER/env.sh
O=checkpoints/momentum_mech
WAVE_BETA=1 WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/wave4500_user_ncap1_4090.txt
WAVE_BETA=1 WAVE_NCAP=2 WAVE_PFUN=tanh WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/wave4500_user_ncap2_4090.txt
WAVE_PFUN=tanh WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/wave4500_tanh_b097_4090.txt
