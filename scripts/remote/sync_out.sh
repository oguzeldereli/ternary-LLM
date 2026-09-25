#!/usr/bin/env bash
# Remote side (runs in tmux next to the queue): every 60 s mirror small run files (logs, metrics)
# from scratch to ~/ternary-sync/runs, and copy each new/updated checkpoint ONCE to
# ~/ternary-sync/outbox. The laptop pulls both and deletes pulled checkpoints from the outbox, so
# home only holds checkpoints in transit. Originals stay on scratch (resumable during the booking).
set -u
S=/scratch0/$USER
R=$S/runs; H=$HOME/ternary-sync
mkdir -p "$S/.synced" "$H/runs" "$H/outbox"
while true; do
  rsync -a --exclude '*.pt' --exclude '*.tmp' "$R/" "$H/runs/"
  find "$R" -name '*.pt' -type f | while read -r f; do
    key=$S/.synced/$(printf '%s' "$f" | md5sum | cut -c1-16)-$(stat -c %Y "$f")
    [ -e "$key" ] && continue
    rel=${f#$R/}; dest=$H/outbox/$rel
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest.part" && mv "$dest.part" "$dest" && touch "$key"
  done
  sleep 60
done
