#!/usr/bin/env bash
# 4090: swing tests at nola_lab @4500, 25% of coordinates, one after another:
#   rate062  control for the sign gate: the old rule at the gate's flip count (103k vs 165k per step: x0.62 rate)
#   grav05 / grav08  asymmetric gravity (measurement only): momentum decays 0.5 / 0.8 where the batch opposes it
#   gunit_b08  flip chance in absolute gradient units (fixed divisor from the 0.97 scale) with memory 0.8
cd "$HOME/ternary-LLM" || exit 1
source /scratch0/$USER/env.sh
O=checkpoints/momentum_mech; mkdir -p $O
w() { local out=$1; shift; env "$@" WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $O/$out; }
w wave4500_rate062_4090.txt WAVE_RATE_MULT=0.62
w wave4500_grav05_4090.txt WAVE_GRAV=0.5
w wave4500_gunit_b08_4090.txt WAVE_BETA=0.8 WAVE_NORM=fixed WAVE_GM_BETA=0.97
w wave4500_grav08_4090.txt WAVE_GRAV=0.8
