#!/usr/bin/env bash
# Lab PC: run the lines of a list file one by one ("NAME train-flags"), re-reading the file before each run so the
# list can be changed while it runs. Waits for the given process pattern first.   runner_lab.sh LIST [WAITPAT]
L=$HOME/ternary-LLM/scripts/night/$1; W=${2:-}
[ -n "$W" ] && while pgrep -u "$USER" -f "$W" >/dev/null; do sleep 30; done
while line=$(grep -m1 -v '^#' "$L" 2>/dev/null) && [ -n "$line" ]; do
  grep -v -x -F "$line" "$L" > "$L.tmp"; mv "$L.tmp" "$L"
  echo "$(date '+%F %T') START $line" >> $HOME/ternary-sync/lab_queue.log
  bash $HOME/ternary-LLM/scripts/lab/scratch_run.sh $line
done
