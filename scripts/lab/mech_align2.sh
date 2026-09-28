#!/usr/bin/env bash
# Lab PC: alignment BEFORE vs AFTER the flips, per arm, at the trained flip rate and at 1/4 of it (164M snapshots).
# Output line by line: ~/ternary-sync/runs/momentum_mech/align2.txt
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
for r in nola_lab:nola5000 mech_v1_b131:mech_v1_b131 mech_user_b131:mech_user_b131; do
  d=${r%%:*}; src=${r##*:}; mkdir -p checkpoints/$d
  [ -f checkpoints/$d/ckpt_5000.pt ] || cp $HOME/ternary-sync/inbox/$src/ckpt_5000.pt checkpoints/$d/
done
for m in 1.0 0.25; do
  for x in "V0 nola_lab" "V2 nola_lab" "V1 mech_v1_b131" "U1 mech_user_b131"; do
    set -- $x; python -u -m scripts.analysis.mech_align2 $1 $2 5000 $m 2>&1 | grep --line-buffered -v Warn
  done
done > $HOME/ternary-sync/runs/momentum_mech/align2.txt
