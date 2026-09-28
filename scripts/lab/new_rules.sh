#!/usr/bin/env bash
# Lab PC: one of the three new training rules as a branch of nola_lab at 131M -> 300M, no look-ahead.
#   new_rules.sh select      online-learned flip selector (propose at 2x the rate, keep the best half)
#   new_rules.sh adaptrate   flip rate adapted to keep momentum predictive of the next gradient
#   new_rules.sh multibeta   momenta with decay 0.8 / 0.95 / 0.99; per layer the best predictor flips
set -u
L=$HOME/ternary-LLM/scripts/lab
case $1 in
  select)    bash $L/branch_run.sh select_b131 --select ;;
  adaptrate) bash $L/branch_run.sh adaptrate_b131 --adapt_rate ;;
  multibeta) bash $L/branch_run.sh multibeta_b131 --multibeta 0.8,0.95,0.99 ;;
esac
