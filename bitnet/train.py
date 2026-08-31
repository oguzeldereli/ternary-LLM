"""Train the master-free ternary BitNet on a flat uint16 token stream.

Data format (nanoGPT-style): a single binary file of uint16 token ids for train
and another for val. Point TrainConfig.data_path / val_path at them.

The BitLinear matrices update themselves inside their backward hooks (no master
weights, no optimizer state tensors). Only the small float tail (embeddings +
norms) uses a real optimizer here.
"""
from __future__ import annotations
import os
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
from .flip import (build_flip_transformer, build_stateless_transformer,
                   build_kernel_transformer, apply_flips)


def get_batch(data, bs, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (bs,))
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
    for _ in range(tc.eval_iters):
        x, y = get_batch(data, tc.batch_size, tc.seq_len, device)
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
                    choices=["kernel", "evidence", "flip", "stateless", "latent"],
                    help="kernel=stateless flips, Triton GEMM; evidence=kernel + "
                         "2-bit per-weight counter; flip=predicted flips + int8 "
                         "evidence (torch); stateless=zero-accumulator (torch); "
                         "latent=int8 master-free latent")
    ap.add_argument("--theta", type=float, default=24.0, help="flip fire threshold")
    ap.add_argument("--rate", type=float, default=2e-2, help="stateless flip rate")
    ap.add_argument("--ev_bits", type=int, default=2,
                    help="evidence mode: counter bit-depth (2,3,4...)")
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc.seed)
    print(mc.report())

    if args.mode == "kernel":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate)
    elif args.mode == "evidence":
        model = build_kernel_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                         rate=args.rate, evidence=True,
                                         ev_bits=args.ev_bits)
    elif args.mode == "flip":
        model, _ = build_flip_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                          theta=args.theta)
    elif args.mode == "stateless":
        model = build_stateless_transformer(mc, grad_checkpoint=tc.grad_checkpoint,
                                            rate=args.rate)
    else:
        model = BitTransformer(mc, grad_checkpoint=tc.grad_checkpoint)
    model = model.to(device)
    model.train()

    # float tail (embeddings + norms + flip predictor): bf16 params + 8-bit Adam
    from .opt8 import Adam8bit
    for p in model.float_tail_parameters():
        p.data = p.data.to(torch.bfloat16)
    tail = model.float_tail_parameters()
    tail_opt = Adam8bit(tail, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                        weight_decay=tc.weight_decay)

    STATE.momentum = args.momentum
    STATE.weight_decay = tc.weight_decay

    train_data = np.memmap(tc.data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(tc.val_path, dtype=np.uint16, mode="r") if os.path.exists(tc.val_path) else train_data

    os.makedirs(tc.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")

    def save_ckpt(step):
        # atomic: write tmp then rename, so a kill mid-save never corrupts ckpt.pt
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": tail_opt.state_dict(),
                    "cfg": mc, "step": step, "mode": args.mode}, tmp)
        os.replace(tmp, ckpt_path)

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if "opt" in blob:
            tail_opt.load_state_dict(blob["opt"])
        start_step = int(blob.get("step", 0)) + 1
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

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
    for step in range(start_step, tc.max_steps):
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
        lr = lr_at(step, tc)
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
            x, y = get_batch(train_data, tc.batch_size, tc.seq_len, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.grad_accum).backward()  # triggers BitLinear fused updates
            last_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(tail, tc.grad_clip)
        tail_opt.step()
        n_flips = apply_flips(model) if args.mode == "flip" else 0

        if step % tc.log_interval == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            extra = f"| flips {n_flips:>7d} " if args.mode == "flip" else ""
            print(f"step {step:6d} | loss {last_loss:6.3f} | lr {lr:.2e} {extra}"
                  f"| {dt:6.1f}s | peakVRAM {mem:.2f}GiB", flush=True)
        if step > 0 and step % tc.eval_interval == 0:
            vl = evaluate(model, val_data, tc, device)
            print(f"  ---- val loss {vl:.3f}", flush=True)
        if time.time() - last_save >= args.save_secs:
            save_ckpt(step)
            last_save = time.time()
            print(f"  ---- checkpoint saved at step {step}", flush=True)

    save_ckpt(tc.max_steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()
