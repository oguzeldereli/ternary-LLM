#!/usr/bin/env bash
# Lab PC side: every 60 s mirror logs/metrics of this machine's runs to ~/ternary-sync/runs and copy each new
# checkpoint once to ~/ternary-sync/outbox (the laptop's pull loop takes both). Run names must not clash with
# runs on the 4090 or the laptop.
set -u
S=/tmp/$USER/tern; R=$S/repo/checkpoints; H=$HOME/ternary-sync
mkdir -p "$S/.synced" "$H/runs" "$H/outbox"
while true; do
  # logs and metrics only: checkpoints (~0.5-1.3 GB each) stay on this PC's local disk, since shipping them through
  # the home folder filled the 10 GB quota; the laptop copies the ones it needs directly from here
  rsync -a --exclude '*.pt' --exclude '*.part' --exclude '*.tmp' "$R/" "$H/runs/"
  sleep 60
done
