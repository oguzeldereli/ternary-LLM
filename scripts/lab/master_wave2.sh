#!/usr/bin/env bash
# Lab PC: master's swing test again, with the move-alignment measure, after the loss-by-position probe.
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
while pgrep -u "$USER" -f "[s]cripts.analysis.loss_by_pos" >/dev/null; do sleep 20; done
cp ~/ternary-LLM/scripts/analysis/master_wave.py scripts/analysis/
python -u -m scripts.analysis.master_wave curve_master 5000 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/master_wave2_5000.txt
