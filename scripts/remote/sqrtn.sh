#!/usr/bin/env bash
# sqrt(n) test of the look-ahead filter at 11M (step 340) and 205M (step 6243), on the checkpoints
# already on scratch. Output: checkpoints/sqrtn_step*.json and checkpoints/sqrtn.log (synced).
set -u
S=/scratch0/$USER; REPO=$HOME/ternary-LLM
source $S/env.sh
cd $REPO
for f in ckpt_340.pt ckpt_6243.pt; do     # scratch is wiped between bookings: take them from the inbox
  [ -f $S/$f ] || { cp $HOME/ternary-sync/inbox/$f $S/$f && rm -f $HOME/ternary-sync/inbox/$f; }
done
for c in "$S/ckpt_340.pt 0.01994" "$S/ckpt_6243.pt 0.00462"; do
  python -m scripts.analysis.filter_sqrt_n $c 64 2>&1 | grep -v Warn | tee -a checkpoints/sqrtn.log
done
echo "$(date '+%F %T') DONE sqrtn"
