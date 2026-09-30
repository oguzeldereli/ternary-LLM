#!/usr/bin/env bash
# Swap the user-rule runs to tighter caps (29 Sep): full runs cap 3 (pintail) and cap 1 (mallard) replace caps 10 / 30;
# the 4090 sweep's remaining arms become caps 1 and 2. Lists are rewritten on the remote home first, then the running
# full runs are stopped, so each machine's runner starts the new line.
K=oguzelde@knuckles.cs.ucl.ac.uk
ssh -o BatchMode=yes $K 'bash -s' <<'EOF'
U="--rc_scale --lr_beta 1.0 --pfun_tanh 10"
echo "user_cap3_s0 $U --mom_ncap 3" > ~/ternary-LLM/scripts/night/pintail.list
echo "user_cap1_s0 $U --mom_ncap 1" > ~/ternary-LLM/scripts/night/mallard.list
printf '%s\n' "usersweep_cap1 $U --mom_ncap 1 --stop_after 1500" "usersweep_cap2 $U --mom_ncap 2 --stop_after 1500" > ~/ternary-LLM/scripts/night/4090.list
cat ~/ternary-LLM/scripts/night/pintail.list ~/ternary-LLM/scripts/night/mallard.list ~/ternary-LLM/scripts/night/4090.list
EOF
for h in pintail mallard; do
  ssh -o BatchMode=yes -J $K oguzelde@$h-l.cs.ucl.ac.uk 'bash -s' <<'EOF'
pkill -TERM -f "out_dir checkpoints/user_cap(10|30)_s0"; sleep 100
echo "$(hostname | cut -d. -f1): $(pgrep -af "[b]itnet.train" | grep -oE "out_dir [a-z0-9_/]+")"
EOF
done
