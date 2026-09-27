#!/usr/bin/env bash
# Lab PC: induction-toy replicates, seeds 4-7 of the three arms the adapter question rests on (all momentum rank 64 +
# look-ahead): no adapter, unfixed additive adapter, adapter with weight decay 0.1. Twelve runs in parallel.
#   setsid nohup bash ~/ternary-LLM/scripts/lab/toy_seeds.sh &
set -u
S=/tmp/$USER/tern; source $S/env.sh; cd $S/repo
for f in toy_ind_v256_train.bin toy_ind_v256_val.bin; do [ -f data/$f ] || cp $HOME/ternary-data/$f data/; done
pgrep -u "$USER" -f "scripts/lab/sync_out.sh" >/dev/null || setsid nohup bash $HOME/ternary-LLM/scripts/lab/sync_out.sh >/dev/null 2>&1 &
run() {
  local n=$1; shift
  STEPS=15000 TOY_TAG=_v256 bash scripts/toy/run.sh $n "$@"
  MAG_KIND=add TOY_V=256 python -m scripts.toy.eval checkpoints/toy/$n kernel 2>&1 | grep "^step" > checkpoints/toy/$n/eval.txt
  echo "$(date '+%F %T') DONE $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/toy/$n/train.log)" >> $HOME/ternary-sync/lab_queue.log
}
LA="--lowrank 64 --lookahead 2 --lookahead_xbatch"
for s in 4 5 6 7; do
  run mom_la_s$s $LA --seed $s &
  run mom_add_la_s$s $LA --lowrank_mag add:4 --seed $s &
  run add_la_wd01_s$s $LA --lowrank_mag add:4 --mag_wd 0.1 --seed $s &
done
wait
