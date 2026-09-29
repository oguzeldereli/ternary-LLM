#!/usr/bin/env bash
# 4090: arm 5 of the friction/undo tests (the script stopped after arm 4): beta 0.97 + norm cap 1x + absolute tanh +
# undo + gate + Adam step, nola_lab @4500, 25% of coordinates.
cd "$HOME/ternary-LLM" || exit 1
source /scratch0/$USER/env.sh
WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_UNDO=1 WAVE_SIG=gate+vnorm:0.99 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_fu5_all_4090.txt
