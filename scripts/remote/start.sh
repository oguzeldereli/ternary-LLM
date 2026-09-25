#!/usr/bin/env bash
# Start the experiment queue inside tmux (survives closing the browser tab; do NOT log out of the
# desktop, that ends the booking session). Attach later with:  /scratch0/$USER/env/bin/tmux attach -t t
set -euo pipefail
S=/scratch0/$USER
"$S/env/bin/tmux" new-session -d -s t "bash -lc 'source $S/env.sh; cd $HOME/ternary-LLM; bash scripts/remote/queue.sh 2>&1 | tee -a checkpoints/remote_queue.log'"
echo "queue started in tmux session t"
