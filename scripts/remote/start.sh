#!/usr/bin/env bash
# Start the experiment queue and the result sync inside tmux (survives closing the browser tab; do
# NOT log out of the desktop, that ends the booking session). Run outputs live on scratch
# (checkpoints -> /scratch0/$USER/runs); sync_out.sh ships them to the laptop through home.
# Attach later with:  /scratch0/$USER/env/bin/tmux attach -t t
set -euo pipefail
S=/scratch0/$USER
REPO=$HOME/ternary-LLM
mkdir -p "$S/runs"
if [ ! -L "$REPO/checkpoints" ]; then
  rmdir "$REPO/checkpoints" 2>/dev/null || mv "$REPO/checkpoints" "$REPO/checkpoints.home.$(date +%s)"
  ln -s "$S/runs" "$REPO/checkpoints"
fi
T=$S/env/bin/tmux
$T new-session -d -s t "bash -lc 'source $S/env.sh; cd $REPO; bash scripts/remote/queue.sh 2>&1 | tee -a checkpoints/remote_queue.log'"
$T new-window -t t "bash $REPO/scripts/remote/sync_out.sh"
echo "queue + sync started in tmux session t"
