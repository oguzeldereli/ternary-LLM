"""Ternary MLP testbed: does a small MLP reproduce the transformer's gap between
stateless flips and BitNet-style master weights? If so, rules can be iterated here in
minutes instead of hours.

Task: next-token prediction on the wiki32k stream from the previous `ctx` tokens
(a Bengio-style neural LM): token embeddings -> concatenate -> ternary MLP -> tied head.
The ternary layers are the same classes the transformer uses:
  --mode master   latent fp32 weights, ternary forward via STE, AdamW (BitNet b1.58)
  --mode flip     stateless packed-ternary flips (KernelTernaryLinear), cosine rate
                  (+ --lookahead N [--xbatch] for the look-ahead filter)
Embeddings and norm gains are fp32 + AdamW in every mode.

  python -m scripts.mlp.mlp_lab --mode master --name mlp_master
  python -m scripts.mlp.mlp_lab --mode flip --name mlp_flip
  python -m scripts.mlp.mlp_lab --mode flip --lookahead 2 --xbatch --name mlp_xb2
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet.model import RMSNorm
from bitnet.master import MasterTernaryLinear
from bitnet.flip import KernelTernaryLinear, set_flip_rate
from bitnet.train import lookahead_step, gpu_temp


class MLPLM(nn.Module):
    def __init__(self, make_linear, vocab=32000, ctx=8, emb=256, hidden=1024, depth=3):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab, emb)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        dims = [ctx * emb] + [hidden] * depth
        self.inp = make_linear(dims[0], hidden)
        self.norms = nn.ModuleList(RMSNorm(hidden) for _ in range(depth))
        self.hid = nn.ModuleList(make_linear(hidden, hidden) for _ in range(depth - 1))
        self.out = make_linear(hidden, emb)
        self.norm_out = RMSNorm(emb)

    def forward(self, idx, targets=None):
        h = self.tok_emb(idx).flatten(1)                          # [B, ctx*emb]
        h = self.inp(h)
        for i, lin in enumerate(self.hid):
            h = h + lin(F.silu(self.norms[i](h)))                 # residual MLP blocks
        h = self.out(F.silu(self.norms[-1](h)))
        logits = F.linear(self.norm_out(h), self.tok_emb.weight)  # tied head
        if targets is None:
            return logits, None
        return None, F.cross_entropy(logits.float(), targets)


def windows(data, n, ctx, gen, device):
    p = torch.randint(0, len(data) - ctx - 1, (n,), generator=gen).numpy()
    w = torch.from_numpy(data[p[:, None] + np.arange(ctx + 1)].astype(np.int64)).to(device)
    return w[:, :ctx], w[:, ctx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["master", "flip"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min_lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rate", type=float, default=0.02, help="peak flip rate (cosine to 0)")
    ap.add_argument("--rate_warmup", type=int, default=10)
    ap.add_argument("--g_ref", type=float, default=3.0)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--xbatch", action="store_true")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--max_temp", type=int, default=86)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs("checkpoints/mlp", exist_ok=True)
    out = open(f"checkpoints/mlp/{a.name}.jsonl", "w")

    torch.manual_seed(a.seed)
    if a.mode == "master":
        mk = lambda i, o: MasterTernaryLinear(i, o)
    else:
        mk = lambda i, o: KernelTernaryLinear(i, o, rate=a.rate, g_ref=a.g_ref, beta=True,
                                              int8=True, dw_mode="dense")
    model = MLPLM(mk, ctx=a.ctx, emb=a.emb, hidden=a.hidden, depth=a.depth).to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and "norm" not in n]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and "norm" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    Ls = [l for l in model.modules() if isinstance(l, KernelTernaryLinear)]
    n_tern = sum(l.N * l.K for l in Ls) or sum(m.weight.numel() for m in model.modules()
                                               if isinstance(m, MasterTernaryLinear))
    print(f"{a.name}: {a.mode}, ternary weights {n_tern / 1e6:.1f}M, "
          f"{a.batch} examples/step x {a.steps} steps", flush=True)

    train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
    val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
    gtr = torch.Generator().manual_seed(1234 + a.seed)
    gla = torch.Generator().manual_seed(4242 + a.seed)
    VAL = [windows(val, a.batch, a.ctx, torch.Generator().manual_seed(99 + i), dev) for i in range(8)]
    extra = (lambda: windows(train, a.batch, a.ctx, gla, dev)) if a.xbatch else None

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        r = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * r))

    def rate_at(s):
        if s < a.rate_warmup:
            return a.rate * s / a.rate_warmup
        r = (s - a.rate_warmup) / max(1, a.steps - a.rate_warmup)
        return a.rate * 0.5 * (1 + math.cos(math.pi * r))

    @torch.no_grad()
    def evaluate():
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = float(np.mean([model(x, y)[1].item() for x, y in VAL]))
        model.train()
        return v

    t0 = time.time()
    for s in range(a.steps):
        if s % 4 == 0:
            t = gpu_temp()
            while t is not None and t >= a.max_temp:
                time.sleep(5); t = gpu_temp()
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        if Ls:
            set_flip_rate(model, rate_at(s))
            for l in Ls: l.capture = bool(a.lookahead)
        x, y = windows(train, a.batch, a.ctx, gtr, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)[1]
        loss.backward()
        info = {}
        if a.lookahead:
            info = lookahead_step(model, x, y, params, dev, s, a.lookahead, 0, extra)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec = {"step": s, "examples": (s + 1) * a.batch, "loss": loss.item(), **info}
        if s % a.eval_every == 0 or s == a.steps - 1:
            rec["val_loss"] = evaluate()
            print(f"step {s:5d} | loss {rec['loss']:.3f} | val {rec['val_loss']:.3f} | "
                  f"{time.time() - t0:6.0f}s", flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
