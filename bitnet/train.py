"""Train the master-free ternary BitNet on a flat uint16 token stream.

Data format (nanoGPT-style): a single binary file of uint16 token ids for train
and another for val. Point TrainConfig.data_path / val_path at them.

The BitLinear matrices update themselves inside their backward hooks (no master
weights, no optimizer state tensors). Only the small float tail (embeddings +
norms) uses a real optimizer here.
"""
from __future__ import annotations
import os
import json
import math
import time
import argparse
import subprocess
import numpy as np
import torch


def gpu_temp():
    """Current GPU temperature in Celsius via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None

from .config import ModelConfig, TrainConfig, PRESETS, DEFAULT_PRESET
from .model import BitTransformer
from .bitlinear import STATE
from .master import build_master_transformer, split_params, MasterTernaryLinear
from .flip import (build_flip_transformer, build_stateless_transformer,
                   build_kernel_transformer, apply_flips, enable_flip_tracking,
                   collect_flip_stats, flip_accumulated, set_flip_accum,
                   set_flip_rate, set_abs_scale, set_err_feedback, set_lockout,
                   reset_lockout, lockout_stats)


def get_batch(data, bs, seq_len, device, gen=None):
    # gen: dedicated RNG so the data order is identical across runs regardless of
    # what else consumes the global RNG (different modes draw differently).
    ix = torch.randint(len(data) - seq_len - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step, tc: TrainConfig):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step > tc.max_steps:
        return tc.min_lr
    r = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def evaluate(model, data, tc, device):
    model.eval()
    STATE.updates_enabled = False
    losses = []
    # fixed windows: same val batches every eval and across runs (comparable).
    gen = torch.Generator().manual_seed(tc.seed + 12345)
    for _ in range(tc.eval_iters):
        x, y = get_batch(data, tc.batch_size, tc.seq_len, device, gen)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="latent mode: 0=stateless SGD+SR; ~0.9=Lion-style int8 momentum")
    ap.add_argument("--mode", default="kernel",
                    choices=["kernel", "evidence", "flip", "stateless", "latent",
                             "master"],
                    help="kernel=stateless flips, Triton GEMM; evidence=kernel + "
                         "2-bit per-weight counter; flip=predicted flips + int8 "
                         "evidence (torch); stateless=zero-accumulator (torch); "
                         "latent=int8 master-free latent; master=latent master "
                         "weights + STE + AdamW (the b->inf baseline)")
    ap.add_argument("--master_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="master mode: latent weight dtype (bf16 rounds away small "
                         "AdamW steps; fp32 is the honest ceiling)")
    ap.add_argument("--theta", type=float, default=24.0, help="flip fire threshold")
    ap.add_argument("--rate", type=float, default=2e-2, help="stateless flip rate")
    ap.add_argument("--ev_bits", type=int, default=2,
                    help="evidence mode: counter bit-depth (2,3,4...)")
    ap.add_argument("--int8", action="store_true",
                    help="kernel/evidence: s8 tensor-core path (exact int32 forward, "
                         "8-bit gy in dx, int8 dw)")
    ap.add_argument("--int8_dx", action="store_true",
                    help="also run grad_x on the s8 kernel (default: bf16 dx kernel)")
    ap.add_argument("--dw_mode", default="dense", choices=["int8", "cublas", "sign", "dense"],
                    help="weight-gradient kernel: int8 tensor cores, sign XNOR outer "
                         "product, or dense torch matmul")
    ap.add_argument("--rate_schedule", default="const",
                    choices=["const", "cosine", "linear", "exp"],
                    help="Arm A: decay the flip rate over the run. exp = "
                         "rate * exp(-step/--rate_tau): halves repeatedly, "
                         "approaches zero asymptotically without reaching it")
    ap.add_argument("--rate_tau", type=float, default=1000.0,
                    help="exp schedule: steps per e-folding")
    ap.add_argument("--abs_scale", action="store_true",
                    help="Arm B: freeze the flip-threshold denominator after "
                         "--calib_steps (absolute scale; flip rate can then fall)")
    ap.add_argument("--calib_steps", type=int, default=200)
    ap.add_argument("--stop_after", type=int, default=None,
                    help="stop after this many steps WITHOUT changing the LR / flip "
                         "schedules (which still span --steps); evaluates on exit")
    ap.add_argument("--flip_lockout", type=int, default=0,
                    help="per-weight lockout: a weight may flip at most once per N "
                         "steps (1 bit/weight). 0 = off")
    ap.add_argument("--lockout_mode", default="once", choices=["once", "noreversal"],
                    help="once = one flip per epoch; noreversal = after the first "
                         "flip only the same direction is allowed")
    ap.add_argument("--err_feedback", action="store_true",
                    help="spatial error feedback: push each layer's unapplied flip "
                         "demand into grad_x (kernel mode, grad_accum 1 only)")
    ap.add_argument("--ef_alpha", type=float, default=1.0,
                    help="mixing weight of the error-feedback term in gy")
    ap.add_argument("--rate_min", type=float, default=0.0,
                    help="floor for --rate_schedule: the schedule decays from --rate "
                         "to this value instead of to zero (keeps flips alive)")
    ap.add_argument("--track_flips", action="store_true",
                    help="log per-layer flip rate + never-changed fraction each step")
    ap.add_argument("--no_beta", action="store_true",
                    help="kernel/evidence: legacy raw-trit forward (no 1/sqrt(K*rho) "
                         "scale). Only for reproducing pre-beta checkpoints.")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--min_lr", type=float, default=None,
                    help="cosine floor (default: lr/10)")
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--eval_iters", type=int, default=None)
    ap.add_argument("--loss_chunk", type=int, default=2048,
                    help="tokens per output-head chunk (peak VRAM knob; math is identical)")
    ap.add_argument("--save_secs", type=float, default=900.0,
                    help="wall-clock seconds between checkpoint saves")
    ap.add_argument("--max_temp", type=int, default=86,
                    help="save + stop if GPU temp (C) reaches this (crash guard)")
    ap.add_argument("--temp_check", type=int, default=4,
                    help="check GPU temp every N steps")
    args = ap.parse_args()

    mc: ModelConfig = PRESETS[args.preset]
    tc = TrainConfig()
    if args.data: tc.data_path = args.data
    if args.val: tc.val_path = args.val
    if args.steps: tc.max_steps = args.steps
    if args.batch_size: tc.batch_size = args.batch_size
    if args.grad_accum is not None: tc.grad_accum = args.grad_accum
    if args.seq_len: mc.max_seq_len = tc.seq_len = args.seq_len
    if args.out_dir: tc.out_dir = args.out_dir
    if args.lr: tc.lr = args.lr
    if args.warmup is not None: tc.warmup_steps = args.warmup
    tc.min_lr = args.min_lr if args.min_lr is not None else tc.lr / 10
    if args.eval_interval: tc.eval_interval = args.eval_interval
    if args.eval_iters: tc.eval_iters = args.eval_iters

    BitTransformer.loss_chunk = args.loss_chunk
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc.seed)
    print(mc.report())
    use_beta = args.mode in ("kernel", "evidence") and not args.no_beta

    if args.mode == "kernel":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate, beta=use_beta,
                                         int8=args.int8, dw_mode=args.dw_mode,
                                         int8_dx=args.int8_dx)
    elif args.mode == "evidence":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate, evidence=True,
                                         ev_bits=args.ev_bits, beta=use_beta,
                                         int8=args.int8, dw_mode=args.dw_mode,
                                         int8_dx=args.int8_dx)
    elif args.mode == "flip":
        model, _ = build_flip_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                          theta=args.theta)
    elif args.mode == "master":
        model = build_master_transformer(
            mc, grad_checkpoint=tc.grad_checkpoint,
            dtype=torch.float32 if args.master_dtype == "fp32" else torch.bfloat16)
    elif args.mode == "stateless":
        model = build_stateless_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                            rate=args.rate)
    else:
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint)
    model = model.to(device)
    model.train()

    if args.mode == "master":
        # the baseline keeps everything in fp32 with a standard AdamW: no bf16
        # rounding anywhere, so the ceiling is not limited by storage precision.
        master, emb, norms = split_params(model)
        tail = master + emb + norms
        tail_opt = torch.optim.AdamW(
            [{"params": master, "weight_decay": tc.weight_decay},
             {"params": emb, "weight_decay": tc.weight_decay},
             {"params": norms, "weight_decay": 0.0}],
            lr=tc.lr, betas=(tc.beta1, tc.beta2))
        print(f"master mode: {sum(p.numel() for p in master)/1e6:.1f}M latent "
              f"({args.master_dtype}) + {sum(p.numel() for p in emb+norms)/1e6:.1f}M tail, "
              f"AdamW fp32 states", flush=True)
    else:
        # float tail (embeddings + norms + flip predictor): bf16 params + 8-bit Adam
        from .opt8 import Adam8bit
        for p in model.float_tail_parameters():
            p.data = p.data.to(torch.bfloat16)
        tail = model.float_tail_parameters()
        tail_opt = Adam8bit(tail, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                            weight_decay=tc.weight_decay)

    STATE.momentum = args.momentum
    STATE.weight_decay = tc.weight_decay

    if args.mode in ("kernel", "evidence") and tc.grad_accum > 1:
        set_flip_accum(model, tc.grad_accum)
        print(f"flip accumulation: one flip per {tc.grad_accum} micro-steps "
              f"({tc.grad_accum * tc.batch_size * tc.seq_len:,} tokens/step)", flush=True)

    if args.flip_lockout > 0:
        n = set_lockout(model, args.flip_lockout, args.lockout_mode)
        print(f"flip lockout: {args.lockout_mode}, epoch {args.flip_lockout} steps "
              f"({n} layers, 2 bits/weight of mask)", flush=True)

    if args.err_feedback:
        if args.mode != "kernel" or tc.grad_accum != 1:
            raise SystemExit("--err_feedback requires --mode kernel and --grad_accum 1")
        n = set_err_feedback(model, True, args.ef_alpha)
        print(f"spatial error feedback on ({n} layers, alpha {args.ef_alpha})", flush=True)

    if args.abs_scale:
        n = set_abs_scale(model, True, args.calib_steps)
        print(f"Arm B: absolute flip scale, frozen after {args.calib_steps} steps "
              f"({n} layers)", flush=True)

    if args.track_flips:
        n = enable_flip_tracking(model)
        print(f"flip tracking on for {n} ternary layers", flush=True)

    sampler = torch.Generator().manual_seed(tc.seed)
    train_data = np.memmap(tc.data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(tc.val_path, dtype=np.uint16, mode="r") if os.path.exists(tc.val_path) else train_data

    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    metrics_path = os.path.join(tc.out_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "a")

    def log_metrics(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    def save_ckpt(step):
        # atomic: write tmp then rename, so a kill mid-save never corrupts ckpt.pt
        tmp = ckpt_path + ".tmp"
        # flip-tracking masks are plain attributes (not state_dict buffers), so they
        # are saved alongside: otherwise "never flipped since init" silently resets
        # to 100% on resume.
        touched = {n: m.touched for n, m in model.named_modules()
                   if hasattr(m, "touched") and m.touched is not None}
        # step the mask started accumulating from: never-flipped is a "since init"
        # number only when this is 0.
        touched_origin = _touched_origin["v"]
        torch.save({"model": model.state_dict(), "opt": tail_opt.state_dict(),
                    "touched": touched, "touched_origin": touched_origin,
                    "cfg": mc, "step": step, "mode": args.mode, "beta": use_beta,
                    "int8": args.int8, "dw_mode": args.dw_mode, "rate_min": args.rate_min,
                    "int8_dx": args.int8_dx, "rate_schedule": args.rate_schedule,
                    "abs_scale": args.abs_scale, "err_feedback": args.err_feedback,
                    "flip_lockout": args.flip_lockout,
                    "lockout_mode": args.lockout_mode,
                    "ef_alpha": args.ef_alpha}, tmp)
        os.replace(tmp, ckpt_path)

    start_step = 0
    _touched_origin = {"v": 0}
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        if blob.get("beta", False) != use_beta:
            raise SystemExit(f"checkpoint beta={blob.get('beta', False)} but run beta="
                             f"{use_beta}: forward differs, refusing to resume")
        model.load_state_dict(blob["model"])
        if "opt" in blob:
            tail_opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0)) + 1
        tmask = blob.get("touched") or {}
        nres = 0
        for n, m in model.named_modules():
            if n in tmask and getattr(m, "touched", None) is not None:
                m.touched.copy_(tmask[n].to(m.touched.device)); nres += 1
        if nres:
            _touched_origin["v"] = int(blob.get("touched_origin", 0))
            print(f"resumed from {ckpt_path} at step {start_step} (flip history for "
                  f"{nres} layers, accumulated since step {_touched_origin['v']})",
                  flush=True)
        else:
            _touched_origin["v"] = start_step
            print(f"resumed from {ckpt_path} at step {start_step}", flush=True)
            if args.track_flips:
                print(f"  WARNING: checkpoint has no flip history -> 'never flipped' "
                      f"counts only from step {start_step}, NOT since init. The true "
                      f"since-init value is <= the last value logged before this "
                      f"resume (it is monotone non-increasing).", flush=True)

    # save on SIGTERM/SIGINT so `kill` (or Ctrl-C) leaves a fresh resumable ckpt
    import signal
    _last = {"step": start_step}
    def _graceful(signum, frame):
        print(f"signal {signum}: saving ckpt at step {_last['step']} ...", flush=True)
        save_ckpt(_last["step"]); print("saved. exiting.", flush=True); raise SystemExit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    t0 = time.time()
    last_save = time.time()
    end_step = tc.max_steps if args.stop_after is None else min(tc.max_steps, args.stop_after)
    for step in range(start_step, end_step):
        _last["step"] = step
        # thermal guard: PAUSE (not stop) while GPU is at/above max_temp; resume
        # in place once it cools. No exit, no save — just wait it out.
        if step % args.temp_check == 0:
            temp = gpu_temp()
            if temp is not None and temp >= args.max_temp:
                print(f"GPU {temp}C >= {args.max_temp}C -> pausing to cool "
                      f"(step {step})", flush=True)
                while temp is not None and temp >= args.max_temp:
                    time.sleep(5)
                    temp = gpu_temp()
                print(f"cooled to {temp}C -> resuming", flush=True)
        if args.flip_lockout > 0 and step % args.flip_lockout == 0:
            reset_lockout(model)
        lr = lr_at(step, tc)
        rate_now = args.rate
        if args.rate_schedule != "const":
            prog = min(1.0, step / max(1, tc.max_steps))
            if args.rate_schedule == "exp":
                # f(step) = rate * exp(-step / tau); never exactly zero
                rate_now = args.rate * math.exp(-step / args.rate_tau)
            else:
                mult = (0.5 * (1 + math.cos(math.pi * prog))
                        if args.rate_schedule == "cosine" else 1.0 - prog)
                rate_now = args.rate_min + (args.rate - args.rate_min) * mult
            set_flip_rate(model, rate_now)
        STATE.lr = lr
        for g in tail_opt.param_groups:
            g["lr"] = lr

        # gradient accumulation: BitLinear hooks apply lr*g/accum each micro-step
        # (SGD is linear, so summing micro-steps approximates one averaged step).
        STATE.updates_enabled = True
        STATE.grad_scale = tc.grad_accum
        tail_opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        for micro in range(tc.grad_accum):
            x, y = get_batch(train_data, tc.batch_size, tc.seq_len, device, sampler)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.grad_accum).backward()  # triggers BitLinear fused updates
            last_loss = loss.item()
        if args.mode in ("kernel", "evidence") and tc.grad_accum > 1:
            flip_accumulated(model)
        torch.nn.utils.clip_grad_norm_(tail, tc.grad_clip)
        tail_opt.step()
        n_flips = apply_flips(model) if args.mode == "flip" else 0

        rec = {"step": step, "loss": last_loss, "lr": lr, "flip_rate_cfg": rate_now,
               "touched_origin": _touched_origin["v"],
               "tokens": (step + 1) * tc.grad_accum * tc.batch_size * tc.seq_len}
        fs = collect_flip_stats(model) if args.track_flips else {}
        if args.flip_lockout > 0 and step % tc.log_interval == 0:
            lf = lockout_stats(model)
            if lf is not None:
                rec["locked_frac"] = lf
        if fs:
            rec.update(flip_frac=fs["flip_frac_total"], never_frac=fs["never_frac_total"],
                       flip_frac_layers=fs["flip_frac"], never_frac_layers=fs["never_frac"],
                       gmean_layers=fs["gmean_layers"],
                       frozen_scale_layers=fs["frozen_scale_layers"])
        log_metrics(rec)

        if step % tc.log_interval == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            extra = f"| flips {n_flips:>7d} " if args.mode == "flip" else ""
            if fs:
                extra += (f"| flip {fs['flip_frac_total']*100:.3f}% "
                          f"| never {fs['never_frac_total']*100:.1f}% ")
            print(f"step {step:6d} | loss {last_loss:6.3f} | lr {lr:.2e} {extra}"
                  f"| {dt:6.1f}s | peakVRAM {mem:.2f}GiB", flush=True)
        if step > 0 and step % tc.eval_interval == 0:
            vl = evaluate(model, val_data, tc, device)
            log_metrics({"step": step, "val_loss": vl, "val_ppl": math.exp(vl)})
            print(f"  ---- val loss {vl:.3f} | ppl {math.exp(vl):.1f}", flush=True)
        if time.time() - last_save >= args.save_secs:
            save_ckpt(step)
            last_save = time.time()
            print(f"  ---- checkpoint saved at step {step}", flush=True)

    save_ckpt(end_step)
    vl = evaluate(model, val_data, tc, device)
    log_metrics({"step": end_step, "val_loss": vl, "val_ppl": math.exp(vl),
                 "final": True})
    print(f"FINAL val loss {vl:.4f} | ppl {math.exp(vl):.2f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
