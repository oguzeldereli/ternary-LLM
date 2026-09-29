"""Does the true gradient oscillate slowly (a wave with a period of tens of steps) while training flips? At a
no-look-ahead snapshot (saved momentum, float tail frozen, plain momentum with the trainer's flip rule and rate),
41 steps of flips with the true gradient (32 batches) measured before every step. Reports:
  autocorrelation  mean cos(T_t, T_t+lag) over t, for lags 1..40: a wave goes negative at half its period
  momentum lag     cos(M_t, T_t+lag): how well the momentum held at step t predicts the gradient lag steps later
  main directions  the 3 principal directions of the 41 gradients and each one's sign over time (swinging back and
                   forth = oscillation along that direction; the number of sign changes gives a rough period)
  WAVE_BETA=1 WAVE_NCAP=1 WAVE_PFUN=tanh   the user's rule as one vector: pure sum, norm cap, absolute tanh flip chance
  WAVE_BETA=1 WAVE_CAP=3 WAVE_NORM=fixed   the user's rule: pure sum (no decay), per-weight cap, fixed divisor
  WAVE_GRAV=0.5        asymmetric gravity: momentum decays with 0.5 where the batch gradient opposes it, BETA elsewhere
  WAVE_SIG=gate | vnorm:0.99 | rowema:0.995   flip signal variants (sign gate; factored-Adam step; per-row speed)
  WAVE_NORM=ema:0.995  the trainer's --speed_ref rule (slow EMA of mean |M|, started at its step-0 value)
  WAVE_NORM=fixed      divide the momentum by its size at step 0 instead of by its current size (the trainer's rule),
                       so fewer weights flip when the momentum shrinks (speed follows velocity, as for a real mass)
  also prints the held-out loss before / after the 41 steps and the flips per step
  WAVE_RATE_MULT=0|0.25 WAVE_BETA=0.8 WAVE_SUB=0.25 WAVE_NORM=fixed python -m scripts.analysis.wave RUN STEP
"""
import sys, math, numpy as np, torch
from scripts.analysis.testbench import Bench, G_REF
from bitnet.kernel import fused_flip
from bitnet.train import get_batch

RUN, ST = sys.argv[1], int(sys.argv[2])
NT, K = 32, int(__import__("os").environ.get("WAVE_K", "41"))
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
if BETA < 1: F = [f * (1 - 0.97) / (1 - BETA) for f in F]   # a sum of gradients has size ~ g / (1 - beta)
# WAVE_SUB < 1 keeps a fixed random share of the coordinates, on the CPU (small GPUs / little RAM)
SUB = float(os.environ.get("WAVE_SUB", "1"))
NTOT = sum(f.numel() for f in F)
IDX = None if SUB >= 1 else torch.randperm(NTOT, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))[:int(SUB * NTOT)]


def keep(v):
    return v.half() if IDX is None else v[IDX].half().cpu()


from bitnet.kernel import unpack_rows
NORM = os.environ.get("WAVE_NORM", "own")
SIG = os.environ.get("WAVE_SIG", "plain"); VN, RR = {}, {}
GCAP = float(os.environ.get("WAVE_GCAP", "0"))   # cap ||M|| <= GCAP x (running mean of the batch gradient norm)
GN = [0.0, 0]
NCAP = float(os.environ.get("WAVE_NCAP", "0"))   # cap on the whole momentum vector: ||M|| <= NCAP x ||M|| at step 0
PFUN = os.environ.get("WAVE_PFUN", "clip")        # tanh: flip chance rate * tanh(|M_ij| / v0), v0 fixed at step 0
CAP = float(os.environ.get("WAVE_CAP", "0"))     # per-weight velocity cap |M_ij| <= CAP x (mean |M| at step 0)
GRAV = float(os.environ.get("WAVE_GRAV", "0"))   # 0 = off; e.g. 0.5: strong pull-back when climbing
GM0 = [f.abs().mean().clamp_min(1e-12) for f in F]
NORM0 = sum(float(f.pow(2).sum()) for f in F) ** 0.5
V0 = [float(x) for x in GM0]      # fixed absolute scale for WAVE_PFUN=tanh
# WAVE_GM_BETA=0.97: the fixed divisor in gradient units taken from the 0.97 momentum (size ~ g / 0.03), so a shorter
# memory (smaller M) flips fewer weights instead of the same number
if os.environ.get("WAVE_GM_BETA"): GM0 = [x * (1 - BETA) / (1 - float(os.environ["WAVE_GM_BETA"])) for x in GM0]


def held():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(np.mean([B.m(x, y)[1].item() for x, y in B.VB]))


L0 = held(); nflip, upshare = [], []
s = batches(5151); Ts, Ms = [], []
for k in range(K):
    W0 = [unpack_rows(l.wpacked, l.K).to(torch.int8) for l in Ls]
    Tf = truth(900 + k); Ts.append(keep(Tf)); Ms.append(keep(torch.cat([f.flatten() for f in F])))
    g = grad(next(s))
    if GRAV:   # asymmetric gravity: where this batch pushes against a weight's momentum, decay it with GRAV, not BETA
        F = [torch.where(f.sign() * x.sign() < 0, GRAV * f, BETA * f) + x for f, x in zip(F, g)]
    else:
        F = [BETA * f + x for f, x in zip(F, g)]
    if GCAP:   # the cap in gradient units: a velocity of at most GCAP gradients (turns in ~GCAP steps)
        GN[0] += sum(float(x.pow(2).sum()) for x in g) ** 0.5; GN[1] += 1
        lim = GCAP * GN[0] / GN[1]; nrm = sum(float(f.pow(2).sum()) for f in F) ** 0.5
        if nrm > lim: F = [f * (lim / nrm) for f in F]
    if NCAP:   # the user's rule: cap the momentum as one vector (all layers together), not per weight
        nrm = sum(float(f.pow(2).sum()) for f in F) ** 0.5
        if nrm > NCAP * NORM0: F = [f * (NCAP * NORM0 / nrm) for f in F]
    if CAP:    # the user's rule with BETA=1: no decay, the loss slows the momentum uphill, the cap bounds its speed
        F = [f.clamp(-CAP * float(GM0[i]), CAP * float(GM0[i])) for i, f in enumerate(F)]
    for i, (l, f) in enumerate(zip(Ls, F)):
        gm = f.abs().mean().clamp_min(1e-12)
        if "vnorm:" in SIG:                    # factored second moment of g (N + K floats): smaller steps where steep
            d = float(SIG.split("vnorm:")[1]); g2 = g[i].pow(2); R, C = g2.mean(1), g2.mean(0)
            if i in VN: R = d * VN[i][0] + (1 - d) * R; C = d * VN[i][1] + (1 - d) * C
            VN[i] = (R, C)
            f = f / (R[:, None] * C[None, :] / R.mean().clamp_min(1e-30)).sqrt().clamp_min(1e-30)
            gm = f.abs().mean().clamp_min(1e-12)
        if SIG.startswith("gate"):             # flip only where this batch's gradient agrees in sign (reacts at once)
            f = f * (f.sign() == g[i].sign())  # (as the trainer: after vnorm; "gate+vnorm:0.99" = both)
        if SIG.startswith("rowema:"):        # per-row speed reference: slow EMA of each row's mean |M|
            d = float(SIG[7:]); rm = f.abs().mean(1).clamp_min(1e-12)
            RR[i] = rm if i not in RR else d * RR[i] + (1 - d) * rm
            f = f * (gm / RR[i])[:, None]
        if NORM == "fixed": gm = GM0[i]
        elif NORM.startswith("ema:"):          # the trainer's --speed_ref: slow EMA of mean |M|
            GM0[i] = float(NORM[4:]) * GM0[i] + (1 - float(NORM[4:])) * gm; gm = GM0[i]
        if PFUN == "tanh":  # absolute units: p = RATE * tanh(|M_ij| / (G_REF v0)); no division by the current size
            gm = V0[i]; f = f.sign() * (G_REF * gm) * torch.tanh(f.abs() / (G_REF * gm))
        fused_flip(l.wpacked, f.contiguous(), RATE, G_REF, 3000 + 131 * k + i, gmean=gm)
        f = None
    # flips made this step, and the share that moved uphill on the true gradient (move has the sign of T)
    up = mv = off = 0
    for l, w in zip(Ls, W0):
        d = (unpack_rows(l.wpacked, l.K).to(torch.int8) - w).flatten(); t = Tf[off:off + d.numel()]; off += d.numel()
        nz = d != 0; mv += int(nz.sum()); up += int(((d.float() * t) > 0)[nz].sum())
    nflip.append(mv); upshare.append(up / max(mv, 1)); del W0, Tf
    if k % 10 == 0: print(f"  step {k} measured", flush=True)
# everything from dot products (stacking 41 full gradients does not fit on the GPU)
def dots(A, Bs):
    return torch.tensor([[float(a.float() @ b.float()) for b in Bs] for a in A], dtype=torch.float64)
G = dots(Ts, Ts); nT = G.diagonal().sqrt(); G = G / nT[:, None] / nT[None, :]    # 41 x 41 cosines
MT = dots(Ms, Ts); nM = torch.tensor([float(m.float().norm()) for m in Ms], dtype=torch.float64)
MT = MT / nM[:, None] / nT[None, :]
L1 = held()
print(f"\nheld-out loss {L0:.4f} -> {L1:.4f} ({L1 - L0:+.4f}) over {K} steps; flips per step: first 5 "
      f"{[round(n / 1e3) for n in nflip[:5]]}k, mean {np.mean(nflip) / 1e3:.0f}k, last 5 {[round(n / 1e3) for n in nflip[-5:]]}k")
print(f"{RUN} @{ST}, norm {NORM}, signal {SIG}, gravity {GRAV}, cap {CAP}, norm cap {NCAP}, grad cap {GCAP}, pfun {PFUN}, rate {RATE:.4f}, beta {BETA}, true gradient from {NT} batches at each of {K} steps")
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

# ---- along the swing: the gradient's and the momentum's component on each main direction, step by step
# (u_j = sum_t c_tj (T_t - mean T), unit; T_s . u_j = (G c_j)_s / sqrt(w_j), M_s/|M_s| . u_j = (MT c_j)_s / sqrt(w_j))
print(f"\nflips that move uphill on the true gradient: mean {100 * np.mean(upshare):.1f}%  "
      f"(first 5 {[round(100 * u) for u in upshare[:5]]}%, last 5 {[round(100 * u) for u in upshare[-5:]]}%)")
wv, Vv = torch.linalg.eigh(Gc)
for j in range(2):
    c = Vv[:, -1 - j]; sw = float(wv[-1 - j]).__abs__() ** 0.5
    tp = (G @ c) / sw; mp = (MT @ c) / sw
    best = max(range(0, 8), key=lambda L: float(torch.corrcoef(torch.stack([tp[:K - L], mp[L:]]))[0, 1]))
    cc = [float(torch.corrcoef(torch.stack([tp[:K - L], mp[L:]]))[0, 1]) for L in range(0, 8)]
    print(f"direction {j + 1}: corr(gradient component at t, momentum component at t+L) for L = 0..7: "
          + " ".join(f"{x:+.2f}" for x in cc) + f"  -> momentum follows ~{best} steps late")
    print("  step:     " + " ".join(f"{t:4d}" for t in range(0, K, 2)))
    print("  gradient: " + " ".join(f"{100 * float(tp[t]):+4.0f}" for t in range(0, K, 2)))
    print("  momentum: " + " ".join(f"{100 * float(mp[t]):+4.0f}" for t in range(0, K, 2)) + "   (x100, cosine with the direction)")
# how much of the momentum lies in the span of the 41 true gradients (the rest points nowhere the gradient goes)
Gi = torch.linalg.pinv(G, rtol=1e-4)
span = [float(MT[t] @ Gi @ MT[t]) for t in range(K)]
print(f"share of the momentum's size (squared) inside the span of the 41 true gradients: mean {np.mean(span):.3f}, "
      f"first {span[0]:.3f}, last {span[-1]:.3f}")
# ---- per weight: when the true gradient on a weight reverses, how many steps until its momentum follows
gsub = torch.Generator().manual_seed(7); n_all = Ts[0].numel()
ci = torch.randperm(n_all, generator=gsub)[:200_000]
Tc_ = torch.stack([t.float().cpu()[ci] for t in Ts]); Mc_ = torch.stack([m.float().cpu()[ci] for m in Ms])
sT, sM = Tc_.sign(), Mc_.sign()
agree = (sT == sM).float().mean(1)
wagree = ((sT == sM).float() * Tc_.abs()).sum(1) / Tc_.abs().sum(1)
print(f"per weight, momentum sign = true-gradient sign: mean {float(agree.mean()):.3f} of weights, "
      f"{float(wagree.mean()):.3f} weighted by |T|")
big = Tc_.abs() > Tc_.abs().median()
delays, cens, ratio = [], 0, []
for t in range(1, K):
    ev = (sT[t] != sT[t - 1]) & big[t] & big[t - 1]            # a real reversal of the true gradient
    idx = ev.nonzero().flatten()
    if idx.numel() == 0: continue
    ratio.append((Mc_[t, idx].abs() / Tc_[t, idx].abs()).median().item())
    d = torch.full((idx.numel(),), -1)
    for dd in range(0, K - t):
        hit = (sM[t + dd, idx] == sT[t, idx]) & (d < 0); d[hit] = dd
    cens += int((d < 0).sum()); delays.append(d[d >= 0])
D = torch.cat(delays).float() if delays else torch.zeros(1)
tot = D.numel() + cens
print(f"true-gradient reversals (both sides above median size): {tot}; momentum already on the new side: "
      f"{100 * float((D == 0).sum()) / tot:.1f}%, follows within 1-3 steps: {100 * float(((D >= 1) & (D <= 3)).sum()) / tot:.1f}%, "
      f"4+ steps: {100 * float((D >= 4).sum()) / tot:.1f}%, never within the window: {100 * cens / tot:.1f}%; "
      f"median delay of those that follow {float(D.median()):.0f} steps")
print(f"at a reversal, |momentum| / |true gradient| on that weight: median {np.median(ratio):.1f} "
      f"(steps to cross zero if each step removes one true gradient's worth)")
