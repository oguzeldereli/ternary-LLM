#!/usr/bin/env bash
# metrics.jsonl (tens of MB per run: per-layer flip fractions every step) is NOT mirrored: the 10 GB home quota filled. Pull it from the machine directly.
# Remote side (runs in tmux next to the queue): every 60 s mirror small run files (logs, metrics) from scratch to
# ~/ternary-sync/runs. Checkpoints (~0.5-1.3 GB each) stay on scratch: shipping them through the home folder
# filled the 10 GB quota twice; the laptop copies the ones it needs directly.
set -u
S=/scratch0/$USER
R=$S/runs; H=$HOME/ternary-sync
mkdir -p "$H/runs"
while true; do
  rsync -a --exclude '*.pt' --exclude '*.part' --exclude '*.tmp' --exclude 'metrics.jsonl' "$R/" "$H/runs/"
  sleep 60
done
