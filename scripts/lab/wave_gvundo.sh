#!/usr/bin/env bash
# Lab PC: gate + Adam step + undo (old flip rule otherwise), the direct comparison with gate + Adam step (-0.092);
# nola_lab @4500, 25%. Output: ~/ternary-sync/runs/momentum_mech/wave4500_gatevnorm_undo_lab.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
while pgrep -u "$USER" -f "[s]cripts.analysis.loss_by_pos" >/dev/null; do sleep 15; done
cp ~/ternary-LLM/scripts/analysis/wave.py ~/ternary-LLM/scripts/analysis/testbench.py scripts/analysis/
WAVE_SIG=gate+vnorm:0.99 WAVE_UNDO=1 WAVE_SUB=0.25 python -u -m scripts.analysis.wave nola_lab 4500 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/wave4500_gatevnorm_undo_lab.txt
