#!/usr/bin/env bash
# One-time setup per booking on a UCL CS GPU workstation (scratch is wiped when the booking ends).
# Builds a user-space env on /scratch0 (micromamba: python 3.13, gcc for Triton, tmux; torch
# 2.13.0+cu126 like the laptop), copies the data from home to scratch and links it into the repo.
#   bash ~/ternary-LLM/scripts/remote/setup.sh
set -euo pipefail
S=/scratch0/$USER
REPO=$HOME/ternary-LLM
mkdir -p "$S"
cd "$S"
if [ ! -x "$S/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$S" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$S/mamba
if [ ! -x "$S/env/bin/python" ]; then
  "$S/bin/micromamba" create -y -p "$S/env" -c conda-forge python=3.13 gcc gxx tmux
fi
"$S/env/bin/pip" install --quiet torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
"$S/env/bin/pip" install --quiet numpy==2.3.5 scipy matplotlib
mkdir -p "$S/data"
for f in wiki32k_train.bin wiki32k_val.bin; do
  [ -f "$S/data/$f" ] || cp "$HOME/ternary-data/$f" "$S/data/$f"
done
rm -f "$REPO/data"; ln -s "$S/data" "$REPO/data"
cat > "$S/env.sh" <<EOT
export PATH=$S/env/bin:\$PATH CC=$S/env/bin/gcc PYTHONPATH=$REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$S/triton-cache
EOT
source "$S/env.sh"
cd "$REPO"
python - <<'P'
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
P
echo "setup ok. next:  bash ~/ternary-LLM/scripts/remote/start.sh"
