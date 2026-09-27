#!/usr/bin/env bash
# Every 10 min: measure induction on new snapshots, redraw the text-induction and live figures.
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  python -m scripts.analysis.induction_track >> checkpoints/induction_track.log 2>&1
  python -m scripts.plots.plot_induction_text >/dev/null 2>&1
  python -m scripts.plots.plot_live >/dev/null 2>&1
  sleep 600
done
