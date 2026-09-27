#!/usr/bin/env bash
# shoveler, after nola_lab: (1) the look-ahead-after-induction branch 164M -> 300M; (2) additive r16 + temperature
# to its planned 200M; (3) additive r16 + weight decay + temperature 131M -> 300M.
set -u
R=$HOME/ternary-LLM/scripts/lab/resume_run.sh
bash $R nola_then_la --lookahead 2 --lookahead_xbatch --snap_every 500
bash $R magadd16_qk_lab --lookahead 2 --lookahead_xbatch --lowrank_mag add:16 --qk_temp --snap_every 1000 --stop_after 6105
bash $R magadd16_wd_qk_lab --lookahead 2 --lookahead_xbatch --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --snap_every 500
