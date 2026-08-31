"""Triton packed-ternary GEMM.

Weight is stored packed 5 trits/byte along the K (input) dimension, one row per
output feature: wpacked[N, K5] uint8, K5 = ceil(K/5). The kernel loads a byte,
decodes 5 trits {-1,0,1} in registers, and accumulates x*trit — so the dense
bf16 weight is never materialized in global memory. y = x @ W^T.

Forward runs on the kernel (also the inference path). Backward re-derives the
weight transiently for one layer at a time (grad_x and the flip), which is
already memory-bounded.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

_P3 = [1, 3, 9, 27, 81]


def pack_rows(w: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} [N,K] -> uint8 [N, ceil(K/5)] base-3, per row along K."""
    N, K = w.shape
    K5 = (K + 4) // 5
    pad = K5 * 5 - K
    x = (w.to(torch.int16) + 1)                       # {0,1,2}
    if pad:
        x = torch.cat([x, x.new_zeros(N, pad)], dim=1)
    x = x.view(N, K5, 5)
    wts = torch.tensor(_P3, dtype=torch.int16, device=w.device)
    return (x * wts).sum(2).to(torch.uint8)           # [N, K5]


def unpack_rows(packed: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [N,K5] -> int8 {-1,0,1} [N,K]."""
    N, K5 = packed.shape
    x = packed.to(torch.int16)
    out = torch.empty(N, K5, 5, dtype=torch.int8, device=packed.device)
    for j in range(5):
        out[:, :, j] = (x % 3).to(torch.int8)
        x //= 3
    return out.view(N, K5 * 5)[:, :K].sub_(1)


def _configs():
    cfgs = []
    for BM in (64, 128):
        for BN in (64, 128):
            for BK5 in (16, 32):
                for w in (4, 8):
                    for s in (2, 3):
                        cfgs.append(triton.Config(
                            {"BM": BM, "BN": BN, "BK5": BK5},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _tern_gemm_kernel(x_ptr, w_ptr, y_ptr, M, N, K, K5,
                      sxm, sxk, swn, swk, sym, syn,
                      BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k5 in range(0, K5, BK5):
        offk5 = k5 + tl.arange(0, BK5)
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                       # [BN, BK5]
        for j in tl.static_range(5):
            trit = ((w // (3 ** j)) % 3 - 1).to(tl.bfloat16)      # [BN, BK5]
            offkx = offk5 * 5 + j                                # x column index
            xj = tl.load(x_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                         mask=(offm[:, None] < M) & (offkx[None, :] < K),
                         other=0.0).to(tl.bfloat16)               # [BM, BK5]
            acc += tl.dot(xj, tl.trans(trit))
    y = acc.to(tl.bfloat16)
    tl.store(y_ptr + offm[:, None] * sym + offn[None, :] * syn, y,
             mask=(offm[:, None] < M) & (offn[None, :] < N))


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _dx_kernel(gy_ptr, w_ptr, gx_ptr, M, N, K, K5,
               sgm, sgn, swn, swk, sxm, sxk,
               BM: tl.constexpr, BN: tl.constexpr, BK5: tl.constexpr):
    # grad_x[m, k] = sum_n gy[m, n] * W[n, k]   (contract over N)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offk5 = pid_k * BK5 + tl.arange(0, BK5)
    a0 = tl.zeros((BM, BK5), tl.float32)
    a1 = tl.zeros((BM, BK5), tl.float32)
    a2 = tl.zeros((BM, BK5), tl.float32)
    a3 = tl.zeros((BM, BK5), tl.float32)
    a4 = tl.zeros((BM, BK5), tl.float32)
    for n0 in range(0, N, BN):
        offn = n0 + tl.arange(0, BN)
        gy = tl.load(gy_ptr + offm[:, None] * sgm + offn[None, :] * sgn,
                     mask=(offm[:, None] < M) & (offn[None, :] < N),
                     other=0.0).to(tl.bfloat16)                  # [BM, BN]
        w = tl.load(w_ptr + offn[:, None] * swn + offk5[None, :] * swk,
                    mask=(offn[:, None] < N) & (offk5[None, :] < K5),
                    other=0).to(tl.int32)                       # [BN, BK5]
        a0 += tl.dot(gy, (((w // 1) % 3 - 1)).to(tl.bfloat16))
        a1 += tl.dot(gy, (((w // 3) % 3 - 1)).to(tl.bfloat16))
        a2 += tl.dot(gy, (((w // 9) % 3 - 1)).to(tl.bfloat16))
        a3 += tl.dot(gy, (((w // 27) % 3 - 1)).to(tl.bfloat16))
        a4 += tl.dot(gy, (((w // 81) % 3 - 1)).to(tl.bfloat16))
    for j in tl.static_range(5):
        aj = a0 * (j == 0) + a1 * (j == 1) + a2 * (j == 2) + a3 * (j == 3) + a4 * (j == 4)
        offkx = offk5 * 5 + j
        tl.store(gx_ptr + offm[:, None] * sxm + offkx[None, :] * sxk,
                 aj.to(tl.bfloat16),
                 mask=(offm[:, None] < M) & (offkx[None, :] < K))


def tern_gemm_dx(gy: torch.Tensor, wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """grad_x = gy @ decode(wpacked), gy [M,N] -> [M,K]. Packed weight, no unpack."""
    M, N = gy.shape
    Nw, K5 = wpacked.shape
    gy = gy.contiguous().to(torch.bfloat16)
    gx = torch.empty(M, K, dtype=torch.bfloat16, device=gy.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(K5, meta["BK5"]))
    _dx_kernel[grid](gy, wpacked, gx, M, N, K, K5,
                     gy.stride(0), gy.stride(1), wpacked.stride(0), wpacked.stride(1),
                     gx.stride(0), gx.stride(1))
    return gx


@triton.jit
def _flip_kernel(w_ptr, g_ptr, N, K, K5, sgn, sgk, gmean, rate, g_ref, seed,
                 BLOCK: tl.constexpr):
    # in-place stochastic flip on packed bytes: decode 5 trits, flip each toward
    # -sign(grad) with prob ~ |grad|, re-encode, write. No dense weight round-trip.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1                                    # current trit {-1,0,1}
        k = k5 * 5 + j
        gmask = mask & (k < K)
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gmask, other=0.0)
        gn = g / gmean
        prob = tl.minimum(tl.abs(gn) / g_ref, 1.0) * rate
        r = tl.rand(seed, idx * 5 + j)
        fire = (r < prob) & gmask
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))         # -sign(grad)
        nt = tl.where(fire, tl.maximum(tl.minimum(t + d, 1), -1), t)
        newb += (nt + 1) * p3
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)


def fused_flip(wpacked: torch.Tensor, grad_w: torch.Tensor, rate: float,
               g_ref: float, seed: int):
    """Apply stochastic ternary flips directly on the packed buffer, in place."""
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    gmean = grad_w.abs().mean().clamp_min(1e-8).item()
    total = N * K5
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _flip_kernel[grid](wpacked, grad_w, N, K, K5,
                       grad_w.stride(0), grad_w.stride(1),
                       gmean, rate, g_ref, seed, BLOCK=BLOCK)


@triton.jit
def _flip_ev_kernel(w_ptr, g_ptr, s_ptr, N, K, K5, sgn, sgk, ssn, gmean, rate,
                    g_ref, seed, smax, BLOCK: tl.constexpr):
    # integrate-and-fire with a signed saturating counter in {-smax..smax}.
    # smax=1 -> 3 levels (2-bit); smax=3 -> 7 levels (3-bit); etc. Each "active"
    # step nudges the counter toward -sign(grad); a flip fires only when the
    # counter is already saturated and pushed further. Deeper = more noise filtering.
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N * K5
    n = idx // K5
    k5 = idx % K5
    b = tl.load(w_ptr + idx, mask=mask, other=0).to(tl.int32)
    newb = tl.zeros((BLOCK,), tl.int32)
    for j in tl.static_range(5):
        p3 = 3 ** j
        t = (b // p3) % 3 - 1                          # current trit
        k = k5 * 5 + j
        gm = mask & (k < K)
        soff = n * ssn + k
        g = tl.load(g_ptr + n * sgn + k * sgk, mask=gm, other=0.0)
        s = tl.load(s_ptr + soff, mask=gm, other=0).to(tl.int32)   # {-1,0,1}
        gn = g / gmean
        prob = tl.minimum(tl.abs(gn) / g_ref, 1.0) * rate
        r = tl.rand(seed, idx * 5 + j)
        active = (r < prob) & gm
        d = tl.where(gn > 0, -1, tl.where(gn < 0, 1, 0))           # -sign(grad)
        s2 = s + tl.where(active, d, 0)
        fire_up = active & (d > 0) & (s == smax)
        fire_dn = active & (d < 0) & (s == -smax)
        newt = tl.where(fire_up, tl.minimum(t + 1, 1),
                        tl.where(fire_dn, tl.maximum(t - 1, -1), t))
        news = tl.where(fire_up | fire_dn, 0,
                        tl.maximum(tl.minimum(s2, smax), -smax))
        tl.store(s_ptr + soff, news.to(tl.int8), mask=gm)
        newb += (newt + 1) * p3
    tl.store(w_ptr + idx, newb.to(tl.uint8), mask=mask)


def fused_flip_ev(wpacked, grad_w, evidence, rate, g_ref, seed, smax=1):
    """Flip with a signed saturating per-weight counter in {-smax..smax}, in place.
    smax=1 -> 2-bit (3 levels), smax=3 -> 3-bit (7 levels). `evidence` is int8 [N,K]."""
    N, K5 = wpacked.shape
    K = grad_w.shape[1]
    gmean = grad_w.abs().mean().clamp_min(1e-8).item()
    total = N * K5
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _flip_ev_kernel[grid](wpacked, grad_w, evidence, N, K, K5,
                          grad_w.stride(0), grad_w.stride(1), evidence.stride(0),
                          gmean, rate, g_ref, seed, smax, BLOCK=BLOCK)


def tern_gemm(x: torch.Tensor, wpacked: torch.Tensor, K: int) -> torch.Tensor:
    """x [M,K] (fp16/bf16) @ decode(wpacked)^T -> y [M,N] fp16."""
    M, Kx = x.shape
    assert Kx == K, (Kx, K)
    N, K5 = wpacked.shape
    x = x.contiguous().to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _tern_gemm_kernel[grid](
        x, wpacked, y, M, N, K, K5,
        x.stride(0), x.stride(1), wpacked.stride(0), wpacked.stride(1),
        y.stride(0), y.stride(1))
    return y
