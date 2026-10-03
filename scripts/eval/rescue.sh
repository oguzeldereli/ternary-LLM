#!/usr/bin/env bash
# Laptop: copy final checkpoints off the lab PCs (their /tmp is wiped at the Monday / Thursday reboots). For each
# "HOST RUN [full]" line of the list file (HOST = a lab PC name, or 4090 for /scratch0 on the booked 4090): on HOST, write eval.pt (ckpt.pt without optimizer / TTF state / flip masks:
# all a final evaluation needs), then copy eval.pt, train.log and metrics.jsonl straight to
# checkpoints/final/RUN/ (and ckpt.pt too with "full"). Skips runs already copied.
#   bash scripts/eval/rescue.sh LIST
set -u
cd "$(dirname "$0")/../.."
J="-o BatchMode=yes -o ConnectTimeout=10 -J oguzelde@knuckles.cs.ucl.ac.uk"
while read -r h r full; do
  [ -z "$h" ] || [ "${h:0:1}" = "#" ] && continue
  D=checkpoints/final/$r; mkdir -p $D
  H=oguzelde@$h-l.cs.ucl.ac.uk; C=/tmp/oguzelde/tern/repo/checkpoints/$r; E=/tmp/oguzelde/tern/env.sh; W=/tmp/oguzelde/tern/repo
  [ "$h" = 4090 ] && { H=oguzelde@beachcomber.cs.ucl.ac.uk; C=/scratch0/oguzelde/runs/$r; E=/scratch0/oguzelde/env.sh; W=~/ternary-LLM; }
  if [ -f $D/eval.pt ]; then
    if [ "${full:-}" = full ] && [ ! -f $D/ckpt.pt ]; then scp -q $J $H:$C/ckpt.pt $D/ && echo "$(date +%T) $r full ckpt added"
    else echo "$(date +%T) $r already there"; fi
    continue
  fi
  ssh $J $H bash -s <<EOS 2>&1 | tail -1
source $E; cd $W
python - $C <<'PY'
import sys, torch
c = torch.load(sys.argv[1] + "/ckpt.pt", map_location="cpu", weights_only=False)
for k in ("opt", "lowrank", "ts_state", "touched"): c.pop(k, None)
torch.save(c, sys.argv[1] + "/eval.pt"); print("slim ok step", c["step"], "mode", c["mode"])
PY
EOS
  scp -q $J $H:$C/{eval.pt,train.log,metrics.jsonl} $D/ || { echo "$(date +%T) $r COPY FAILED"; rm -f $D/eval.pt; continue; }
  [ "${full:-}" = full ] && scp -q $J $H:$C/ckpt.pt $D/
  echo "$(date +%T) $r $(du -sh $D | cut -f1) final $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)"
done < "$1"
echo "rescue done"
