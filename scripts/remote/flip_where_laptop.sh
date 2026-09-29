#!/usr/bin/env bash
# Laptop: where the flips land under each rule, on three checkpoints @5000 (plain, vnorm, gate+vnorm runs).
cd "$(dirname "$0")/../.." || exit 1
for r in rc_s0 vnorm_rc_s0 gatevnorm_rc_s0; do
  python -u -m scripts.analysis.flip_where $r 5000 2>&1 | grep --line-buffered -v Warn > checkpoints/momentum_mech/flip_where_$r.txt
done
