#!/usr/bin/env bash
# Lab PC side: every 60 s mirror logs/metrics of this machine's runs to ~/ternary-sync/runs and copy each new
# checkpoint once to ~/ternary-sync/outbox (the laptop's pull loop takes both). Run names must not clash with
# runs on the 4090 or the laptop.
set -u
S=/tmp/$USER/tern; R=$S/repo/checkpoints; H=$HOME/ternary-sync
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
