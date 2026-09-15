"""Standard BitNet b1.58 training: latent master weights + STE + AdamW.

This is the b -> infinity limit of the k-bit evidence counter, and the ceiling
the master-free flip runs are measured against. The weight stored is a latent
high-precision matrix; the FORWARD PASS quantizes it to ternary (absmean, as in
BitNet b1.58) with a straight-through estimator, so the network that computes the
loss is ternary — only the accumulator behind it is not.

Latent dtype is fp32 by default: in bf16 (7 mantissa bits) an AdamW step of ~1e-4
on a weight of ~0.02 falls below half the representable gap and rounds away, and
the weight-decay term (lr*wd*w ~ 1e-6) always does. That silently cripples the
baseline, which is the one run that must not be crippled.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import BitTransformer
from .bitlinear import _act_quant_ste


def _ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """BitNet b1.58 absmean ternary {-1,0,1}*gamma with straight-through."""
    gamma = w.detach().abs().mean().clamp_min(1e-5)
    t = (w / gamma).round().clamp_(-1, 1) * gamma
    return w + (t - w).detach()


class MasterTernaryLinear(nn.Module):
    """Ternary-in-the-forward linear with a latent master weight (an nn.Parameter)."""

    def __init__(self, in_features: int, out_features: int, act_bits: int = 8,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.K, self.N, self.act_bits = in_features, out_features, act_bits
        w = torch.empty(out_features, in_features, dtype=dtype)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        self.weight = nn.Parameter(w)

    def forward(self, x):
        # under autocast both operands go to bf16, matching the kernel path's math
        return F.linear(_act_quant_ste(x, self.act_bits), _ternary_ste(self.weight))

    @torch.no_grad()
    def ternary_weight(self):
        w = self.weight.float()
        gamma = w.abs().mean().clamp_min(1e-5)
        return (w / gamma).round().clamp_(-1, 1).to(torch.int8), gamma

    def extra_repr(self):
        return f"in={self.K}, out={self.N}, ternary fwd (STE), latent {self.weight.dtype}"


def build_master_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                             dtype: torch.dtype = torch.float32):
    def make_linear(i, o):
        return MasterTernaryLinear(i, o, c.act_bits, dtype=dtype)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


def split_params(model):
    """(latent master weights, embedding/head, norm gains) — for AdamW groups."""
    master = [m.weight for m in model.modules() if isinstance(m, MasterTernaryLinear)]
    ids = {id(p) for p in master}
    emb, norms = [], []
    for n, p in model.named_parameters():
        if id(p) in ids:
            continue
        (norms if n.endswith("norm.weight") else emb).append(p)
    return master, emb, norms
