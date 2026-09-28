#!/usr/bin/env bash
# One wave-test arm at nola_lab @4500, beta 0.97, 25% of coordinates.   bash wave_one.sh NORM TAG
# Output: checkpoints/momentum_mech/wave4500_norm_<NORM>_<TAG>.txt (TAG = machine: the 4090 log sync copies its
# momentum_mech folder onto the laptop's, so the names must differ)
cd "$(dirname "$0")/../.." || exit 1
[ -f /scratch0/$USER/env.sh ] && source /scratch0/$USER/env.sh
mkdir -p checkpoints/momentum_mech
WAVE_NORM=$1 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_norm_${1/:/}_$2.txt
