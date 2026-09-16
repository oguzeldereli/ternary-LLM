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
                     fused_flip_ev, trit_beta, trit_diff, popcount,
                     tern_gemm_i8, tern_gemm_dx_i8, act_quant_i8, dw_int8, dw_sign,
                     dw_cublas, expand_trit_mask)

_FLIP_SEED = [0]


class _KernelTernFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wpacked, layer):
        beta = trit_beta(wpacked, layer.K) if layer.use_beta else None
        if layer.int8:
            # int8 path: activations are already 8-bit, so x @ W^T is exact in int32
            # on the s8 tensor cores, and the saved activation is 1 byte/element.
            xq, xs = act_quant_i8(x.reshape(-1, layer.K), layer.act_bits)
            y = tern_gemm_i8(xq, xs, wpacked, layer.K,
                             float(beta) if beta is not None else 1.0)
            y = y.view(*x.shape[:-1], layer.N)
            ctx.save_for_backward(xq, xs, wpacked)
        else:
            xq = _act_quant_plain(x, layer.act_bits)
            y = tern_gemm(xq.reshape(-1, layer.K), wpacked, layer.K).view(
                *x.shape[:-1], layer.N)
            if beta is not None:
                y = y * beta.to(y.dtype)
            ctx.save_for_backward(x, wpacked)
        ctx.layer = layer
        ctx.beta = beta
        ctx.xshape = x.shape
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        layer = ctx.layer
        beta = ctx.beta
        gyf = gy.reshape(-1, layer.N).contiguous()
        if beta is not None:
            gyf = gyf * beta.to(gyf.dtype)       # y = beta * x@W^T: both grads carry beta
        # weight gradient first: error feedback needs it (and the flip) BEFORE grad_x
        if layer.int8:
            xq, xs, wpacked = ctx.saved_tensors
            if layer.dw_mode == "sign":
                gw = dw_sign(gyf, xq)
            elif layer.dw_mode == "int8":
                gw = dw_int8(gyf, xq, xs)
            elif layer.dw_mode == "cublas":
                gw = dw_cublas(gyf, xq, xs)
            else:
                gw = gyf.transpose(0, 1) @ (xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype))
        else:
            x, wpacked = ctx.saved_tensors
            xq = _act_quant_plain(x, layer.act_bits).reshape(-1, layer.K)
            xs = None
            gw = gyf.transpose(0, 1) @ xq                       # [N,K] dense grad

        if layer.err_feedback and not layer.capture:
            # ---- spatial error feedback -------------------------------------
            # The flip rule samples a discrete move: E[dw] = p * d, but what is
            # applied is fired * d (d = -sign(g)). The difference is the demand this
            # layer failed to satisfy this step -- dominated NOT by the weights that
            # stayed put (p is tiny) but by the ones that fired and so moved a full
            # level instead of p of one. It is pushed to the layers behind us in the
            # same backward pass (a change in x can produce the change in y the
            # weight change would have), so nothing is stored across steps.
            gx = _ef_backward(layer, gyf, xq, xs, gw, wpacked)
            del gw
            return gx.view(ctx.xshape).to(gy.dtype), None, None

        gx = _grad_x(layer, gyf, wpacked).view(ctx.xshape)
        if layer.capture:
            layer.gw = gw.detach().float()      # diagnostics (diag_snr.py): keep grad, no flip
            return gx.to(gy.dtype), None, None
        if layer.accum_steps > 1:
            # gradient accumulation: sum dL/dW over micro-steps and flip ONCE per
            # optimizer step (flip_accumulated), so "tokens per step" means the same
            # thing for the flip rule as it does for the tail optimizer.
            if layer.gw_accum is None:
                layer.gw_accum = torch.zeros(layer.N, layer.K, dtype=torch.float32,
                                             device=gw.device)
            layer.gw_accum += gw.float()
            layer.accum_seen += 1
            del gw
            return gx.to(gy.dtype), None, None
        _FLIP_SEED[0] += 1
        before = wpacked.clone() if layer.track else None
        gm = layer._flip_scale(gw)
        if layer.evidence is not None:
            fused_flip_ev(wpacked, gw, layer.evidence, layer.rate, layer.g_ref,
                          _FLIP_SEED[0], smax=layer.ev_max, gmean=gm)
        else:
            fused_flip(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0], gmean=gm)
        if before is not None:
            layer._record_flips(before, wpacked)
        del gw
        return gx.to(gy.dtype), None, None


def _grad_x(layer, gyf, wpacked):
    if layer.int8 and layer.int8_dx:
        return tern_gemm_dx_i8(gyf, wpacked, layer.K)
    return tern_gemm_dx(gyf, wpacked, layer.K)


@torch.no_grad()
def _ef_backward(layer, gyf, xq, xs, gw, wpacked):
    """Flip first, measure what the flip actually did, fold the unmet demand into
    gy, then compute grad_x once from the corrected gy. Returns grad_x [M, K].

    Only valid for the plain stochastic flip (kernel mode, no evidence counter, no
    gradient accumulation): p below is that rule's exact flip probability.
    """
    gm = layer._flip_scale(gw)          # the denominator the kernel will use (live or frozen)
    gn = gw.float() / gm
    d = -torch.sign(gn)                                     # descent direction
    p = (gn.abs() / layer.g_ref).clamp_(0, 1) * layer.rate  # P(flip) per weight
    del gn

    before = wpacked.clone()
    _FLIP_SEED[0] += 1
    fused_flip(wpacked, gw, layer.rate, layer.g_ref, _FLIP_SEED[0], gmean=gm)
    fired = expand_trit_mask(trit_diff(before, wpacked), layer.K)
    if layer.track:
        layer._record_flips(before, wpacked)

    # residual demand in weight space, mapped into y space: r = x @ R^T
    R = ((p - fired.to(p.dtype)) * d).to(gyf.dtype)         # [N, K]
    del p, d, fired
    xf = xq.to(gyf.dtype) if xs is None else xq.to(gyf.dtype) * xs[:, None].to(gyf.dtype)
    r = xf @ R.transpose(0, 1)                               # [M, N]
    del R, xf
    # r is the output change this layer still OWES (a desired delta-y). Descending
    # gy moves y by -gy, so asking earlier layers to deliver +r means SUBTRACTING
    # it from gy. (Adding it asks them to amplify the overshoot instead.)
    # Scale-match to gy: the flip rule normalizes by mean|g|, so only the SHAPE of
    # the added signal matters and alpha is a dimensionless mixing weight.
    r = r * (gyf.abs().mean() / r.abs().mean().clamp_min(1e-12))
    gyf = gyf - layer.ef_alpha * r
    del r
    # grad_x must be taken at the weights the FORWARD used, i.e. before this flip
    return _grad_x(layer, gyf, before)


class KernelTernaryLinear(nn.Module):
    """Pure-ternary linear, forward on the Triton packed-ternary GEMM (weights
    stay packed, never materialized dense). Row-packed 5-trits/byte along K."""
    def __init__(self, in_features, out_features, act_bits: int = 8,
                 rate: float = 2e-3, g_ref: float = 3.0, evidence: bool = False,
                 ev_bits: int = 2, beta: bool = False, int8: bool = False,
                 dw_mode: str = "int8", int8_dx: bool = False):
        super().__init__()
        self.use_beta = beta      # scale output by 1/sqrt(K*rho); False = legacy raw trits
        self.int8 = int8          # s8 tensor-core path (fwd exact, dx 8-bit gy)
        self.int8_dx = int8_dx    # use the s8 dx kernel (else the bf16 one)
        self.dw_mode = dw_mode    # int8 | cublas | sign (XNOR) | dense
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
        # --- diagnostics: capture dL/dW instead of flipping (see diag_snr.py) ---
        self.capture = False
        self.gw = None
        # --- gradient accumulation for the flip rule ----------------------------
        self.accum_steps = 1      # micro-steps per optimizer step
        self.accum_seen = 0
        self.gw_accum = None
        # --- flip threshold scale (Arm B: frozen absolute scale) ----------------
        # abs_scale=False -> denominator is this step's mean|g| (relative, cannot
        # converge: if every gradient shrinks, the ratio is unchanged and the same
        # fraction keeps flipping). True -> average mean|g| over the first
        # calib_steps then FREEZE, so the flip rate falls on its own.
        self.abs_scale = False
        self.calib_steps = 200
        self.calib_sum = 0.0
        self.calib_n = 0
        self.last_gmean = 0.0
        self.register_buffer("frozen_scale", torch.zeros(()), persistent=True)
        # --- spatial error feedback (no persistent state; see _ef_backward) -----
        self.err_feedback = False
        self.ef_alpha = 1.0
        # --- flip instrumentation (off unless enable_tracking() is called) ------
        self.track = False
        self.touched = None      # uint8 [N,K5] sticky mask: trit ever changed
        self.flips = None        # 0-dim int64: trit changes since last reset

    @torch.no_grad()
    def _flip_scale(self, grad_w):
        """Denominator for the flip threshold; see abs_scale."""
        g = grad_w.abs().mean().clamp_min(1e-8).item()
        self.last_gmean = g
        if not self.abs_scale:
            return g
        if float(self.frozen_scale) > 0.0:
            return float(self.frozen_scale)
        self.calib_sum += g
        self.calib_n += 1
        if self.calib_n >= self.calib_steps:
            self.frozen_scale.fill_(self.calib_sum / self.calib_n)
        return g

    def enable_tracking(self):
        """Track per-step flip count and which trits have ever changed since init.

        Costs one packed-size uint8 mask (~1.6 bit/weight) plus a packed clone per
        backward. Counts trit CHANGES per fused_flip call, so with grad_accum > 1
        the per-step number sums the micro-steps.
        """
        self.track = True
        dev = self.wpacked.device
        self.touched = torch.zeros_like(self.wpacked)
        self.flips = torch.zeros((), dtype=torch.int64, device=dev)

    @torch.no_grad()
    def _record_flips(self, before, after):
        d = trit_diff(before, after)
        self.touched |= d
        self.flips += popcount(d)

    @torch.no_grad()
    def flip_stats(self):
        """(flips since reset, never-changed trits) as 0-dim int64 GPU tensors."""
        pad = self.N * (self.wpacked.shape[1] * 5 - self.K)
        never = (self.N * self.K + pad) - popcount(self.touched) - pad
        return self.flips, never

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
        if self.use_beta:
            return w, trit_beta(self.wpacked, self.K).item()
        scale = w.to(torch.float32).abs().mean().clamp_min(1e-5)
        return w, scale


def build_kernel_transformer(c: ModelConfig, grad_checkpoint: bool = True,
                             rate: float = 2e-3, evidence: bool = False,
                             ev_bits: int = 2, beta: bool = False,
                             int8: bool = False, dw_mode: str = "int8",
                             int8_dx: bool = False):
    def make_linear(i, o):
        return KernelTernaryLinear(i, o, c.act_bits, rate=rate, evidence=evidence,
                                   ev_bits=ev_bits, beta=beta, int8=int8,
                                   dw_mode=dw_mode, int8_dx=int8_dx)
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


@torch.no_grad()
def flip_accumulated(model):
    """Apply one flip per ternary layer from the accumulated gradient. No-op for
    layers running at accum_steps == 1 (they flip inside backward)."""
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear) and m.accum_steps > 1 and m.accum_seen:
            g = m.gw_accum / m.accum_seen
            _FLIP_SEED[0] += 1
            before = m.wpacked.clone() if m.track else None
            gm = m._flip_scale(g)
            if m.evidence is not None:
                fused_flip_ev(m.wpacked, g, m.evidence, m.rate, m.g_ref,
                              _FLIP_SEED[0], smax=m.ev_max, gmean=gm)
            else:
                fused_flip(m.wpacked, g, m.rate, m.g_ref, _FLIP_SEED[0], gmean=gm)
            if before is not None:
                m._record_flips(before, m.wpacked)
            m.gw_accum.zero_()
            m.accum_seen = 0


def set_err_feedback(model, enabled: bool, alpha: float = 1.0):
    """Spatial error feedback: push each layer's unapplied flip demand into grad_x."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            if enabled and (m.evidence is not None or m.accum_steps > 1):
                raise ValueError("error feedback needs plain stochastic flips: "
                                 "kernel mode, no evidence counter, grad_accum 1")
            m.err_feedback = enabled; m.ef_alpha = alpha; n += 1
    return n


def set_flip_rate(model, rate: float):
    """Arm A: open-loop flip-rate schedule."""
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.rate = rate


def set_abs_scale(model, enabled: bool, calib_steps: int = 200):
    """Arm B: freeze the flip-threshold denominator after calibration."""
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.abs_scale = enabled; m.calib_steps = calib_steps; n += 1
    return n


def set_flip_accum(model, steps: int):
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.accum_steps = steps; n += 1
    return n


def enable_flip_tracking(model):
    n = 0
    for m in model.modules():
        if isinstance(m, KernelTernaryLinear):
            m.enable_tracking(); n += 1
    return n


@torch.no_grad()
def collect_flip_stats(model, reset: bool = True):
    """Per-layer flip stats for one step. One host sync for the whole model."""
    names, flips, never, numels = [], [], [], []
    gmeans, frozen = [], []
    for name, m in model.named_modules():
        if isinstance(m, KernelTernaryLinear) and m.track:
            f, nv = m.flip_stats()
            names.append(name); flips.append(f); never.append(nv)
            numels.append(m.N * m.K)
            gmeans.append(m.last_gmean); frozen.append(float(m.frozen_scale))
    if not names:
        return {}
    f = torch.stack(flips).cpu().tolist()          # single sync
    nv = torch.stack(never).cpu().tolist()
    if reset:
        for m in model.modules():
            if isinstance(m, KernelTernaryLinear) and m.track:
                m.flips.zero_()
    tot = sum(numels)
    return {
        "layers": names,
        "flip_frac": [fi / n for fi, n in zip(f, numels)],
        "never_frac": [ni / n for ni, n in zip(nv, numels)],
        "flip_frac_total": sum(f) / tot,
        "never_frac_total": sum(nv) / tot,
        "gmean_layers": gmeans,
        "frozen_scale_layers": frozen,
    }


def apply_flips(model) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, TernaryFlipLinear):
            total += m.flip_step()
    return total
