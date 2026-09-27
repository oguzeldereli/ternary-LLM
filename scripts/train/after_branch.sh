#!/usr/bin/env bash
# Laptop, after the nola_then_la branch reaches 164M: ship its checkpoint to the lab queue (shoveler extends it to
# 300M), then diagnose the additive adapter on text: the failed no-look-ahead run vs the look-ahead one.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
K=oguzelde@knuckles.cs.ucl.ac.uk
until grep -q "DONE nola_then_la" checkpoints/nola_then_la.out 2>/dev/null; do sleep 30; done
ssh -o BatchMode=yes $K "mkdir -p ternary-sync/inbox/nola_then_la"
rsync -a checkpoints/nola_then_la/ckpt.pt checkpoints/nola_then_la/train.log checkpoints/nola_then_la/metrics.jsonl \
  $K:ternary-sync/inbox/nola_then_la/ && ssh -o BatchMode=yes $K "touch ternary-sync/inbox/nola_then_la/READY"
{
  echo "== no look-ahead + additive r16 (nola_add16)"
  MAG_KIND=add python -m scripts.analysis.adapter_diag checkpoints/nola_add16 500 1000 2000 3000
  echo "== look-ahead + additive r16 + temperature (magadd16_qk)"
  MAG_KIND=add python -m scripts.analysis.adapter_diag checkpoints/magadd16_qk 1000 3000 5000
} 2>&1 | grep -v Warn > checkpoints/adapter_diag.txt
echo "$(date '+%F %T') adapter diagnosis done" >> checkpoints/adapter_diag.txt
