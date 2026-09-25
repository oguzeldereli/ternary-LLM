#!/usr/bin/env bash
# Laptop side: every 2 min pull remote run logs/metrics into checkpoints/ and move checkpoints out of
# the home outbox (deleted there once transferred). Run: nohup scripts/remote/pull_loop.sh &
set -u
cd "$(dirname "$0")/../.."
K=oguzelde@knuckles.cs.ucl.ac.uk
while true; do
  rsync -a "$K:ternary-sync/runs/" checkpoints/ 2>>checkpoints/remote_pull.err
  rsync -a --remove-source-files --exclude '*.part' "$K:ternary-sync/outbox/" checkpoints/ 2>>checkpoints/remote_pull.err
  sleep 120
done
