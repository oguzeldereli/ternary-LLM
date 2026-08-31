"""Ternary weights trained by a learned FLIP PREDICTOR.

No latent weight, no int8 master. Each weight is a pure trit w in {-1,0,1}.
Training never nudges a continuous value; instead a small shared network — the
*flip predictor* — looks at the backprop signal and predicts the discrete
transition for each weight: shift down (e.g. 0 -> -1), stay, or shift up
(e.g. 0 -> +1). Only whole-level flips ever happen.

Mechanism per BitLinear-shaped layer
------------------------------------
State (buffers only, no nn.Parameter):
  weight   int8 in {-1,0,1}          -> packs to ~1.58 bit at deploy
  evidence int8                       -> integrate-and-fire accumulator of grad

Forward:
  feat   = [weight, evidence]                     (per element)
  logits = predictor(feat)  -> P(down), P(stay), P(up)
  delta  = P(up) - P(down)  in [-1,1]             (predicted flip direction)
  w_used = weight  (value), with a straight-through path so the task loss
           trains the predictor: grad to delta == step * dL/dweight.
  y = ternary_matmul(quantized_x, w_used)

Backward (hook on w_used):
  g = dL/dweight. Descent wants weight to move opposite g, so
  evidence += quantize(-g). (This is the accumulated backprop pressure.)

Flip step (called once per optimizer step, after backward):
  A weight flips when the integrated evidence crosses a threshold AND the
  predictor agrees on the direction:
     evidence >= +theta and predictor delta > 0 and weight < 1  -> weight += 1
     evidence <= -theta and predictor delta < 0 and weight > -1 -> weight -= 1
  evidence resets on flip. The predictor is the learned veto/accelerator that
  decides whether a given backprop step should actually shift the bit.

The predictor is ONE tiny shared network (~a few hundred params) trained by the
task loss for the whole model. Persistent per-weight memory: weight + evidence.
Set evidence_bits low, or disable the predictor veto, to trade stability for
memory.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import BitTransformer
from .bitlinear import _act_quant_ste
from .packed import TritStore, unpack5, pack5


def _act_quant_plain(x, bits=8):
    """Detached 8-bit activation quant (no autograd graph); used inside the
    custom Function where the straight-through path is handled manually."""
    qp = 2 ** (bits - 1) - 1
    scale = qp / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp_(-qp - 1, qp) / scale


class _PackedStatelessFn(torch.autograd.Function):
    """Memory-bounded ternary linear. Forward saves only the input activation and
    the packed weight (tiny) — NOT the dense bf16 weight. Backward re-unpacks
    transiently, computes the weight gradient, applies the stochastic flip to the
    packed buffer in place, then frees everything. Peak weight-space memory is one
    layer, independent of depth."""
    @staticmethod
    def forward(ctx, x, packed, layer):
        w = unpack5(packed, layer.numel).view(layer.shape).to(x.dtype)
        xq = _act_quant_plain(x, layer.act_bits)
        y = F.linear(xq, w)
        ctx.layer = layer
        ctx.save_for_backward(x, packed)
        del w
        return y

    @staticmethod
    def backward(ctx, gy):
        x, packed = ctx.saved_tensors
        layer = ctx.layer
        w = unpack5(packed, layer.numel).view(layer.shape).to(gy.dtype)
        gx = torch.matmul(gy, w)                      # STE through act-quant
        del w
        xq = _act_quant_plain(x, layer.act_bits)
        gw = torch.matmul(gy.reshape(-1, gy.shape[-1]).transpose(0, 1),
                          xq.reshape(-1, xq.shape[-1]))   # [out, in]
        layer._flip_from_grad(gw, packed)             # mutate packed in place
        del gw
        return gx, None, None


class FlipPredictor(nn.Module):
    """Shared net: per-weight features -> {down, stay, up} logits."""
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        # bias 'stay' initially so an untrained predictor is cautious
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0]))

    def forward(self, feat):          # feat [..., 2] -> [..., 3]
        return self.net(feat)


class TernaryFlipLinear(nn.Module):
    def __init__(self, in_features, out_features, predictor: FlipPredictor,
                 act_bits: int = 8, theta: float = 24.0, e_gain: float = 4.0,
                 step: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.theta = theta          # evidence needed to fire a flip
        self.e_gain = e_gain        # how fast evidence integrates grad
        self.step = step            # STE scale that trains the predictor
        # wrap predictor in a list so it is NOT registered as a submodule here
        # (it is owned once by the top model, trained once).
        self._pred = [predictor]

        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)   # {-1,0,1}
        # weight stored packed 5-trits/byte (~1.6 bit); evidence kept int8
        self.wstore = TritStore(self, "weight_packed", weight)
        self.register_buffer("evidence", torch.zeros_like(weight))
        self._delta = None          # last predicted flip direction (detached)

    @property
    def predictor(self):
        return self._pred[0]

    def forward(self, x):
        wf = self.wstore.unpack().to(x.dtype)           # transient unpack
        ef = self.evidence.to(x.dtype) / 127.0
        feat = torch.stack([wf, ef], dim=-1)            # [out,in,2]
        logits = self.predictor(feat)
        p = logits.softmax(dim=-1)
        delta = p[..., 2] - p[..., 0]                   # [-1,1]

        # value == pure ternary weight; gradient flows to predictor via delta
        w_used = wf + self.step * delta - (self.step * delta).detach()

        x_q = _act_quant_ste(x, self.act_bits)
        y = F.linear(x_q, w_used)

        if w_used.requires_grad:
            w_used.register_hook(self._integrate)
        self._delta = delta.detach()
        return y

    def _integrate(self, g):
        """Accumulate backprop pressure into the int8 evidence register."""
        with torch.no_grad():
            gn = g / g.detach().abs().mean().clamp_min(1e-8)   # normalize scale
            inc = torch.round(-gn * self.e_gain)               # descent direction
            e = self.evidence.to(torch.float32) + inc
            self.evidence.copy_(e.clamp_(-127, 127).to(torch.int8))
        return g   # pass gradient through unchanged

    @torch.no_grad()
    def flip_step(self):
        """Apply predicted flips. Returns number of weights flipped."""
        if self._delta is None:
            return 0
        e = self.evidence.to(torch.float32)
        w = self.wstore.unpack()
        d = self._delta
        up = (e >= self.theta) & (d > 0) & (w < 1)
        dn = (e <= -self.theta) & (d < 0) & (w > -1)
        w[up] += 1
        w[dn] -= 1
        fired = up | dn
        if fired.any():
            self.wstore.write(w)                         # repack changed trits
        self.evidence[fired] = 0
        return int(fired.sum().item())

    @torch.no_grad()
    def ternary_weight(self):
        """Deploy view: {-1,0,1} trits + one absmean scale."""
        w = self.wstore.unpack()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"ternary flip-predicted, theta={self.theta}")


class StatelessFlipLinear(nn.Module):
    """Zero-accumulator ternary layer. Only per-weight state is the trit itself
    (packs to ~1.58 bit / 6.8 GiB at 37B). No evidence register, no predictor.

    The gradient noise is averaged in TIME instead of in a register: each step,
    each weight flips one level toward the descent direction with a small
    probability that scales with this step's gradient magnitude. Over many steps
    the expected drift follows the true gradient; a near-zero-gradient weight
    flips rarely and symmetrically, so it stays put on average.

    Honest: slower and noisier than the evidence version, convergence at LLM
    scale is unproven. This is the literal 'no accumulation at all' option.
    """
    def __init__(self, in_features, out_features, act_bits: int = 8,
                 rate: float = 2e-3, g_ref: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.rate = rate            # base flip probability ceiling
        self.g_ref = g_ref          # grad (in units of mean|g|) that saturates prob
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)
        # only state: weight, packed 5-trits/byte (~1.6 bit / 7.2 GB at 37B)
        self.wstore = TritStore(self, "weight_packed", weight)
        self.numel = weight.numel()
        self.shape = tuple(weight.shape)

    def forward(self, x):
        return _PackedStatelessFn.apply(x, self.weight_packed, self)

    @torch.no_grad()
    def _flip_from_grad(self, g, packed):
        """Stochastic flip applied straight into the packed buffer (in place)."""
        gn = g / g.abs().mean().clamp_min(1e-8)
        prob = (gn.abs() / self.g_ref).clamp_(0, 1) * self.rate
        fire = torch.rand_like(prob) < prob
        if not fire.any():
            return
        w = unpack5(packed, self.numel).view(self.shape)
        direction = -torch.sign(gn).to(torch.int8)                # descend
        w[fire] = (w + direction).clamp_(-1, 1)[fire]
        packed.copy_(pack5(w))

    @torch.no_grad()
    def ternary_weight(self):
        w = self.wstore.unpack()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"stateless stochastic flip, rate={self.rate}")


def build_stateless_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                                rate: float = 2e-3):
    def make_linear(i, o):
        return StatelessFlipLinear(i, o, c.act_bits, rate=rate)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


# ---- Triton-kernel stateless layer (fast path) ----------------------------
from .kernel import (pack_rows, unpack_rows, tern_gemm, tern_gemm_dx, fused_flip,
                     fused_flip_ev)

_FLIP_SEED = [0]


class _KernelTernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        xq = _act_quant_plain(x, layer.act_bits)
        xf = xq.reshape(-1, layer.K)
        y = tern_gemm(xf, wpacked, layer.K).view(*x.shape[:-1], layer.N)
        ctx.layer = layer
        ctx.save_for_backward(x, wpacked)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, wpacked = ctx.saved_tensors
        layer = ctx.layer
        gyf = gy.reshape(-1, layer.N).contiguous()
        gx = tern_gemm_dx(gyf, wpacked, layer.K).view_as(x)     # grad_x, kernel
        xq = _act_quant_plain(x, layer.act_bits).reshape(-1, layer.K)
        gw = gyf.transpose(0, 1) @ xq                           # [N,K] dense grad
        _FLIP_SEED[0] += 1
        if layer.evidence is not None:
            fused_flip_ev(wpacked, gw, layer.evidence, layer.rate, layer.g_ref,
                          _FLIP_SEED[0], smax=layer.ev_max)      # k-bit counter
        else:
            fused_flip(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0])
        del gw
        return gx.to(gy.dtype), None, None


class KernelTernaryLinear(nn.Module):
    """Pure-ternary linear, forward on the Triton packed-ternary GEMM (weights
    stay packed, never materialized dense). Row-packed 5-trits/byte along K."""
    def __init__(self, in_features, out_features, act_bits: int = 8,
                 rate: float = 2e-3, g_ref: float = 3.0, evidence: bool = False,
                 ev_bits: int = 2):
        super().__init__()
        self.K = in_features
        self.N = out_features
        self.act_bits = act_bits
        self.rate = rate
        self.g_ref = g_ref
        self.ev_max = 2 ** (ev_bits - 1) - 1      # 2-bit->1, 3-bit->3, 4-bit->7
        w = torch.empty(out_features, in_features)
        nn.init.normal_(w, std=1.0 / math.sqrt(in_features))
        scale = w.abs().mean().clamp_min(1e-5)
        weight = (w / scale).round().clamp_(-1, 1).to(torch.int8)
        self.register_buffer("wpacked", pack_rows(weight))       # [N, ceil(K/5)]
        # 3-level (2-bit) per-weight counter, {-1,0,1}; int8 storage at this scale
        if evidence:
            self.register_buffer("evidence",
                                 torch.zeros(out_features, in_features, dtype=torch.int8))
        else:
            self.evidence = None

    def forward(self, x):
        return _KernelTernFn.apply(x, self.wpacked, self)

    @torch.no_grad()
    def _flip_from_grad(self, g, wpacked):
        gn = g / g.abs().mean().clamp_min(1e-8)
        prob = (gn.abs() / self.g_ref).clamp_(0, 1) * self.rate
        fire = torch.rand_like(prob) < prob
        if not fire.any():
            return
        w = unpack_rows(wpacked, self.K)                         # [N,K]
        direction = -torch.sign(gn).to(torch.int8)
        w[fire] = (w + direction).clamp_(-1, 1)[fire]
        wpacked.copy_(pack_rows(w))

    @torch.no_grad()
    def ternary_weight(self):
        w = unpack_rows(self.wpacked, self.K)
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale


def build_kernel_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                             rate: float = 2e-3, evidence: bool = False,
                             ev_bits: int = 2):
    def make_linear(i, o):
        return KernelTernaryLinear(i, o, c.act_bits, rate=rate, evidence=evidence,
                                   ev_bits=ev_bits)
    return BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)


def build_flip_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                           theta: float = 24.0, e_gain: float = 4.0,
                           step: float = 0.05, hidden: int = 16):
    """BitTransformer whose linears are flip-predicted ternary layers, sharing
    one predictor (owned by the model so it trains once)."""
    predictor = FlipPredictor(hidden)

    def make_linear(i, o):
        return TernaryFlipLinear(i, o, predictor, c.act_bits, theta, e_gain, step)

    model = BitTransformer(c, grad_checkpoint=grad_checkpoint, make_linear=make_linear)
    model.add_module("flip_predictor", predictor)   # single owner -> trains once
    return model, predictor


def apply_flips(model) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, TernaryFlipLinear):
            total += m.flip_step()
    return total
