#!/usr/bin/env bash
# master vs stateless flips vs cross-batch look-ahead on the ternary MLP LM
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=${STEPS:-6000}
python -m scripts.mlp.mlp_lab --mode master --name mlp_master --steps $S > checkpoints/mlp/mlp_master.log 2>&1
python -m scripts.mlp.mlp_lab --mode flip --name mlp_flip --steps $S > checkpoints/mlp/mlp_flip.log 2>&1
python -m scripts.mlp.mlp_lab --mode flip --lookahead 2 --xbatch --name mlp_xb2 --steps $S > checkpoints/mlp/mlp_xb2.log 2>&1
echo ALLDONE >> checkpoints/mlp/mlp_xb2.log
