#!/usr/bin/env bash
# Myriad: user-space env on Scratch (micromamba: python 3.13, gcc for Triton; torch 2.13 cu126 like the other machines)
set -euo pipefail
S=$HOME/Scratch/tern; mkdir -p $S; cd $S
[ -x $S/bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba
export MAMBA_ROOT_PREFIX=$S/mamba
[ -x $S/env/bin/python ] || $S/bin/micromamba create -y -q -p $S/env -c conda-forge python=3.13 gcc gxx
$S/env/bin/python -c "import torch" 2>/dev/null || $S/env/bin/pip install --quiet --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
$S/env/bin/python -c "import scipy, matplotlib" 2>/dev/null || $S/env/bin/pip install --quiet --no-cache-dir numpy==2.3.5 scipy matplotlib
cat > $S/env.sh <<EOT
export PATH=$S/env/bin:\$PATH CC=$S/env/bin/gcc PYTHONPATH=$HOME/Scratch/ternary-LLM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$S/triton-cache
EOT
source $S/env.sh; python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
echo "myriad setup ok"
