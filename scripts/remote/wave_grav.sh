#!/usr/bin/env bash
# 4090: swing test with asymmetric gravity (0.5, 0.8) at nola_lab @4500, 25% of coordinates, after the running wave.
cd "$(dirname "$0")/../.." || exit 1
source /scratch0/$USER/env.sh
while pgrep -u "$USER" -f "[w]ave_one.sh" >/dev/null; do sleep 20; done
for gr in 0.5 0.8; do
  WAVE_GRAV=$gr WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/wave4500_grav${gr/./}_4090.txt
done
