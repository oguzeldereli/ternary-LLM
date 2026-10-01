#!/bin/bash -l
# Myriad: one 110M run (preset small), 300M tokens of wiki32k, 1 A100. KIND = master (master weights, plus any
# --m_* restriction) or recipe (our rule: dry + spend, rank 512 unless overridden). Resubmit the same line to resume.
#   qsub -N NAME scripts/myriad/run_110m.sh NAME KIND [extra train flags]      (from ~/Scratch/ternary-LLM)
#$ -l h_rt=10:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=LUV
#$ -pe smp 4
#$ -l tmpfs=20G
#$ -wd /home/zcabogu/Scratch/ternary-LLM
#$ -o /home/zcabogu/Scratch/ternary-LLM/logs/
#$ -e /home/zcabogu/Scratch/ternary-LLM/logs/
set -u
S=$HOME/Scratch/tern
RUN="apptainer exec --nv -B /myriadfs -B $HOME/Scratch -B $TMPDIR $S/ubuntu24.sif bash -c"
N=$1; KIND=$2; shift 2; D=checkpoints/$N; mkdir -p $D
hostname >> $D/train.log; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $D/train.log
C="--preset small --data data/wiki32k_train.bin --val data/wiki32k_val.bin --seq_len 2048 --batch_size 16 --grad_accum 1
   --steps 9155 --warmup 305 --lr 1.5e-3 --min_lr 1.5e-4 --eval_interval 250 --eval_iters 30 --save_secs 1800
   --snap_every 1000 --track_flips"
if [ "$KIND" = master ]; then
  F="$C --mode master --master_dtype fp32"
else
  F="$C --mode kernel --rate_schedule cosine --rate 0.0 --rate_peak 0.02 --rate_warmup 30 --g_ref 3.0 --int8
     --dw_mode dense --tail_fp32 --lookahead 0 --lowrank 512 --ckpt_skip 2 --rc_scale --lr_gate --lr_vnorm 0.99
     --lowrank_mag add:16 --mag_wd 0.1 --qk_temp --lr_beta 1 --dry_vec 0.0303 --spend 3"
fi
F=$(echo $F "$@")
[ -f $D/ckpt.pt ] && F="$F --resume"
$RUN "source $S/env.sh; cd $HOME/Scratch/ternary-LLM; python -m bitnet.train $F --out_dir $D" >> $D/train.log 2>&1
echo "$(date '+%F %T') job end $(grep -oE 'FINAL val loss [0-9.]+' $D/train.log | tail -1)" >> $D/train.log
