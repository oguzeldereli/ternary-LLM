#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m scripts.mlp.mlp_lab --mode flip --rate 1.0 --steps 6000 --name mlp_flip_r1 > checkpoints/mlp/mlp_flip_r1.log 2>&1
python -m scripts.mlp.mlp_lab --mode flip --rate 1.0 --steps 6000 --lookahead 2 --xbatch --name mlp_xb2_r1 > checkpoints/mlp/mlp_xb2_r1.log 2>&1
echo ALLDONE >> checkpoints/mlp/mlp_xb2_r1.log
