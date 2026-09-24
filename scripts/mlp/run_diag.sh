#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "mlp_lab.*mlp_lowrank256_xb2" >/dev/null; do sleep 10; done
python -m scripts.mlp.mlp_lab --mode flip --rate 0.0 --name mlp_frozen > checkpoints/mlp/mlp_frozen.log 2>&1
GEOM_OUT=checkpoints/mlp/geometry_proj.json python -m scripts.mlp.geometry master flip xb2 > checkpoints/mlp/geometry_proj.log 2>&1
echo ALLDONE >> checkpoints/mlp/geometry_proj.log
