"""Does the true gradient oscillate slowly (a wave with a period of tens of steps) while training flips? At a
no-look-ahead snapshot (saved momentum, float tail frozen, plain momentum with the trainer's flip rule and rate),
41 steps of flips with the true gradient (32 batches) measured before every step. Reports:
  autocorrelation  mean cos(T_t, T_t+lag) over t, for lags 1..40: a wave goes negative at half its period
  momentum lag     cos(M_t, T_t+lag): how well the momentum held at step t predicts the gradient lag steps later
  main directions  the 3 principal directions of the 41 gradients and each one's sign over time (swinging back and
                   forth = oscillation along that direction; the number of sign changes gives a rough period)
  WAVE_RATE_MULT=0|0.25 WAVE_BETA=0.8 WAVE_SUB=0.25 python -m scripts.analysis.wave RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from bitnet.kernel import fused_flip
from bitnet.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
NT, K = 32, 41
import os
BETA = float(os.environ.get("WAVE_BETA", "0.97"))   # momentum memory; the saved momentum is rescaled to it
train = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
prog = min(1.0, (ST - 30) / (9155 - 30)); RATE = 0.02 * 0.5 * (1 + math.cos(math.pi * prog))
import os
RATE *= float(os.environ.get("WAVE_RATE_MULT", "1"))   # 0 = frozen weights (control), 0.25 = quarter rate
B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names


def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [get_batch(train, 16, 2048, "cuda", g)]


def grad(b):
    g = B.grad(b); return [g[n] for n in names]


def truth(seed):
    s = batches(seed); acc = None
    for _ in range(NT):
        g = grad(next(s)); acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return torch.cat([(a / NT).flatten() for a in acc])


F = [(U.cuda().float() @ V.cuda().float().T) for U, V in B.b["lowrank"]]
F = [f * (1 - 0.97) / (1 - BETA) for f in F]              # a sum of gradients has size ~ g / (1 - beta)
# WAVE_SUB < 1 keeps a fixed random share of the coordinates, on the CPU (small GPUs / little RAM)
SUB = float(os.environ.get("WAVE_SUB", "1"))
NTOT = sum(f.numel() for f in F)
IDX = None if SUB >= 1 else torch.randperm(NTOT, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))[:int(SUB * NTOT)]


def keep(v):
    return v.half() if IDX is None else v[IDX].half().cpu()


s = batches(5151); Ts, Ms = [], []
for k in range(K):
    Ts.append(keep(truth(900 + k))); Ms.append(keep(torch.cat([f.flatten() for f in F])))
    g = grad(next(s)); F = [BETA * f + x for f, x in zip(F, g)]
    for i, (l, f) in enumerate(zip(Ls, F)):
        fused_flip(l.wpacked, f.contiguous(), RATE, G_REF, 3000 + 131 * k + i, gmean=f.abs().mean().clamp_min(1e-12))
    if k % 10 == 0: print(f"  step {k} measured", flush=True)
# everything from dot products (stacking 41 full gradients does not fit on the GPU)
def dots(A, Bs):
    return torch.tensor([[float(a.float() @ b.float()) for b in Bs] for a in A], dtype=torch.float64)
G = dots(Ts, Ts); nT = G.diagonal().sqrt(); G = G / nT[:, None] / nT[None, :]    # 41 x 41 cosines
MT = dots(Ms, Ts); nM = torch.tensor([float(m.float().norm()) for m in Ms], dtype=torch.float64)
MT = MT / nM[:, None] / nT[None, :]
print(f"\n{RUN} @{ST}, rate {RATE:.4f}, beta {BETA}, true gradient from {NT} batches at each of {K} steps")
print("lag  mean cos(T_t, T_t+lag)   mean cos(M_t, T_t+lag)")
for lag in (0, 1, 2, 3, 5, 8, 10, 13, 16, 20, 25, 30, 35, 40):
    c = float(torch.diagonal(G, lag).mean())
    cm = float(torch.diagonal(MT, lag).mean())
    print(f"{lag:3d}  {c:+22.3f}   {cm:+22.3f}")
# principal directions of the (unit) gradient sequence, centered, via the Gram matrix
Gc = G - G.mean(0, keepdim=True) - G.mean(1, keepdim=True) + G.mean()
w, V = torch.linalg.eigh(Gc)
share = (w / w.sum()).flip(0)
print(f"\nmean direction share of the energy: {float(G.mean()):.3f}")
for j in range(3):
    v = V[:, -1 - j]
    signs = "".join("+" if x > 0 else "-" for x in v.tolist())
    changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    print(f"direction {j + 1}: {100 * float(share[j]):.1f}% of the varying part; sign over the 41 steps: {signs} "
          f"({changes} sign changes)")
