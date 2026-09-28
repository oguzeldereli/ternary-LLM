#!/usr/bin/env bash
# 4090, next to the filler: the rate x1/4 half of the before/after alignment test, in reverse arm order (goosander
# does the same list forwards; they meet in the middle). Output: ~/ternary-sync/runs/momentum_mech/align2_4090.txt
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
for r in nola_lab:nola5000 mech_v1_b131:mech_v1_b131 mech_user_b131:mech_user_b131; do
  d=${r%%:*}; src=${r##*:}; mkdir -p checkpoints/$d
  [ -f checkpoints/$d/ckpt_5000.pt ] || cp $HOME/ternary-sync/inbox/$src/ckpt_5000.pt checkpoints/$d/
done
for x in "U1 mech_user_b131" "V1 mech_v1_b131" "V2 nola_lab" "V0 nola_lab"; do
  set -- $x; python -u -m scripts.analysis.mech_align2 $1 $2 5000 0.25 2>&1 | grep --line-buffered -v Warn
done > $HOME/ternary-sync/runs/momentum_mech/align2_4090.txt
echo "$(date '+%F %T') DONE align2_quarter"
