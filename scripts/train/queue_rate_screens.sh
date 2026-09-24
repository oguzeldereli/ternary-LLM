#!/usr/bin/env bash
# Flip-rate screens with cross-batch look-ahead (2 passes), base = la_xb2 (val 5.0021).
set -u
cd "$(dirname "$0")/../.."
for r in 0.004 0.125; do
  n=la_xb2_r${r}
  grep -q "FINAL val" checkpoints/$n/train.log 2>/dev/null && continue
  rm -rf checkpoints/$n
  echo "$(date '+%F %T') START $n" >> checkpoints/rate_screens.log
  scripts/train/screen_10m.sh $n --lookahead 2 --lookahead_xbatch --rate_peak $r
  echo "$(date '+%F %T') END $n $(grep -oE 'FINAL val loss [0-9.]+' checkpoints/$n/train.log)" >> checkpoints/rate_screens.log
done
