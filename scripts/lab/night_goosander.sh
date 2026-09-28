#!/usr/bin/env bash
# goosander, night of 27-28 Sep: after the full mechanism test, pause la_sched98 (checkpoint saved), run the user's
# design without the move correction (gain 0: target point + rotation) as a branch 131->300M, then resume la_sched98.
set -u
L=$HOME/ternary-LLM/scripts/lab
while pgrep -u "$USER" -f "scripts.analysis.momentum_mech" >/dev/null; do sleep 30; done
pkill -TERM -u "$USER" -f "out_dir checkpoints/la_sched98$"
bash $L/mech_run.sh mech_user_g0_b131 user branch 9156 --mech_gain 0
bash $L/la_sched98.sh
