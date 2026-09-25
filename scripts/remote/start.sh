#!/usr/bin/env bash
# Start a queue script (default queue.sh) and, if not already running, the result sync, inside tmux
# session t (survives closing the browser tab; do NOT log out of the desktop, that ends the booking).
# Run outputs live on scratch (checkpoints -> /scratch0/$USER/runs); sync_out.sh ships them to the
# laptop through home.   bash ~/ternary-LLM/scripts/remote/start.sh [bench.sh]
# Attach:  /scratch0/$USER/env/bin/tmux attach -t t      (Ctrl-b d to detach)
set -euo pipefail
S=/scratch0/$USER
REPO=$HOME/ternary-LLM
Q=${1:-queue.sh}
mkdir -p "$S/runs"
if [ ! -L "$REPO/checkpoints" ]; then
  rmdir "$REPO/checkpoints" 2>/dev/null || mv "$REPO/checkpoints" "$REPO/checkpoints.home.$(date +%s)"
  ln -s "$S/runs" "$REPO/checkpoints"
fi
T=$S/env/bin/tmux
CMD="bash -lc 'source $S/env.sh; cd $REPO; bash scripts/remote/$Q 2>&1 | tee -a checkpoints/remote_queue.log'"
if $T has-session -t t 2>/dev/null; then
  $T new-window -t t: "$CMD"
else
  $T new-session -d -s t "$CMD"
fi
pgrep -u "$USER" -f "scripts/remote/sync_out.sh" >/dev/null || $T new-window -t t: "bash $REPO/scripts/remote/sync_out.sh"
echo "started $Q in tmux session t (sync running)"
