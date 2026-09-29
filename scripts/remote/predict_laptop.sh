#!/usr/bin/env bash
# Laptop: 41-step swing test with the importance split, the never-turning analysis and next-gradient prediction:
# old rule, then the user's rule + undo (beta 0.97, norm cap 1x, absolute tanh, undo). 25% of coordinates.
cd "$(dirname "$0")/../.." || exit 1
O=checkpoints/momentum_mech
WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/predict_plain_laptop.txt
WAVE_NCAP=1 WAVE_PFUN=tanh WAVE_UNDO=1 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/predict_user_undo_laptop.txt
