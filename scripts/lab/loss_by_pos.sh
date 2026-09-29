#!/usr/bin/env bash
# Lab PC: loss by context position + copy gain for the given RUN:STEP:KIND checkpoints (local /tmp repo).
# Output: ~/ternary-sync/runs/momentum_mech/lbp_<host>.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh || exit 1; cd $S/repo || exit 1
cp ~/ternary-LLM/scripts/analysis/loss_by_pos.py ~/ternary-LLM/scripts/analysis/induction_heads.py scripts/analysis/
python -u -m scripts.analysis.loss_by_pos "$@" 2>&1 | grep --line-buffered -v Warn > $HOME/ternary-sync/runs/momentum_mech/lbp_$(hostname -s).txt
