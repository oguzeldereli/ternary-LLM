#!/usr/bin/env bash
# shoveler, night of 27-28 Sep: mechanisms first (V1 branch 131->300M, V1 from scratch to 131M), then the
# unfinished look-ahead r16 runs (to 200M, then the weight-decay one to 300M).
set -u
L=$HOME/ternary-LLM/scripts/lab
bash $L/mech_run.sh mech_v1_b131 v1 branch 9156
bash $L/mech_run.sh mech_v1_s0 v1 scratch 4001
bash $L/resume_run.sh magadd16_qk_lab --lookahead 2 --lookahead_xbatch --lowrank_mag add:16 --qk_temp --snap_every 1000 --stop_after 6105
bash $L/resume_run.sh magadd16_wd_qk_lab --lookahead 2 --lookahead_xbatch --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --snap_every 500
