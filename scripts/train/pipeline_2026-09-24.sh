#!/usr/bin/env bash
# Unattended pipeline (night of 2026-09-24). Stages are idempotent: a finished stage is
# skipped, so after a machine reset just run this script again.
#   1. la_xb1   10M screen: cross-batch look-ahead, 1 pass
#   2. la_xb2   10M screen: cross-batch look-ahead, 2 passes
#   3. efficiency test (per-step signal vs curvature) on existing checkpoints
#   4. full 300M run of the best of {same-batch look-ahead (la_fast_fp32tail, val 5.1883),
#      la_xb1, la_xb2}, resumed if interrupted
# Thermal: pause at 85 C, resume at 80 C, checked every step (the machine has hard-reset
# under sustained full load at 80-86 C).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.
LOG=checkpoints/pipeline_2026-09-24.log
GUARD="--max_temp 85 --resume_temp 80 --temp_check 1"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
final_val() { grep -oE "FINAL val loss [0-9.]+" "checkpoints/$1/train.log" 2>/dev/null | tail -1 | awk '{print $4}'; }

screen() {  # name, extra flags
  local name=$1; shift
  if [ -n "$(final_val "$name")" ]; then say "$name already done (val $(final_val "$name"))"; return; fi
  rm -rf "checkpoints/$name"
  say "START $name $*"
  scripts/train/screen_10m.sh "$name" $GUARD "$@" || say "$name exited with $?"
  say "END $name val $(final_val "$name")"
}

screen la_xb1 --lookahead 1 --lookahead_xbatch
screen la_xb2 --lookahead 2 --lookahead_xbatch

if [ ! -f checkpoints/efficiency_test_v2.done ]; then
  say "START efficiency test"
  CK="checkpoints/la_fastramp_lr15/ckpt_100.pt checkpoints/la_fastramp_lr15/ckpt_200.pt
      checkpoints/la_fastramp_lr15/ckpt_300.pt checkpoints/la_rwarm/ckpt_400.pt
      checkpoints/la_rwarm/ckpt_500.pt checkpoints/la_rwarm/ckpt_700.pt"
  { FRAC=0.01 python -m scripts.analysis.efficiency_test $CK
    FRAC=1    python -m scripts.analysis.efficiency_test $CK; } \
    > checkpoints/efficiency_test_v2.log 2>&1 && touch checkpoints/efficiency_test_v2.done
  say "END efficiency test"
fi

# pick the winner by final val (lower is better); baseline = same-batch look-ahead
best=base; bestv=5.1883; flags=""
for pair in "la_xb1:--lookahead 1 --lookahead_xbatch" "la_xb2:--lookahead 2 --lookahead_xbatch"; do
  n=${pair%%:*}; f=${pair#*:}; v=$(final_val "$n")
  [ -z "$v" ] && continue
  if python -c "import sys; sys.exit(0 if $v < $bestv - 0.005 else 1)"; then best=$n; bestv=$v; flags=$f; fi
done
if [ -f checkpoints/overnight_full/winner ]; then
  best=$(cut -d' ' -f1 checkpoints/overnight_full/winner); flags=$(cut -d' ' -f2- checkpoints/overnight_full/winner)
else
  mkdir -p checkpoints/overnight_full; echo "$best $flags" > checkpoints/overnight_full/winner
fi
say "full run config: $best ($flags), screen val $bestv"

for attempt in 1 2 3 4 5; do
  if [ -f checkpoints/overnight_full/done ]; then say "full run done"; break; fi
  R=""; [ -f checkpoints/overnight_full/ckpt.pt ] && R="--resume"
  say "START full run attempt $attempt $R"
  if scripts/train/full_run.sh overnight_full $GUARD $flags $R && grep -q "^done" checkpoints/overnight_full/train.log; then
    touch checkpoints/overnight_full/done
  else
    say "full run exited early (attempt $attempt)"; sleep 60
  fi
done
say "pipeline finished"
