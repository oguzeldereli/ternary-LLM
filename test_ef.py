"""Spatial error feedback: plumbing and invariants.

1. alpha = 0 is bit-identical to EF off (same flips, same grad_x)
2. EF never changes WHICH weights flip at this layer (flip happens before the
   feedback is folded in), only the grad_x sent to earlier layers
3. alpha > 0 changes grad_x, stays finite, and the added term is gy-scaled
4. guards: evidence mode / grad accumulation are rejected
"""
import torch
from bitnet import flip as F
from bitnet.flip import KernelTernaryLinear, set_err_feedback, build_kernel_transformer
from bitnet.config import ModelConfig

dev = "cuda"
torch.manual_seed(0)
K, N, B, T = 768, 2048, 4, 256
x0 = torch.randn(B, T, K, device=dev, dtype=torch.bfloat16)
gy = torch.randn(B, T, N, device=dev, dtype=torch.bfloat16)


def run(int8, ef, alpha, seed=1000):
    torch.manual_seed(1)
    lay = KernelTernaryLinear(K, N, beta=True, rate=2e-2, int8=int8).to(dev)
    lay.err_feedback, lay.ef_alpha = ef, alpha
    x = x0.clone().requires_grad_(True)
    y = lay(x)
    F._FLIP_SEED[0] = seed
    y.backward(gy)
    return x.grad.float(), lay.wpacked.clone()


for int8 in (True, False):
    g_off, w_off = run(int8, False, 0.0)
    g_a0, w_a0 = run(int8, True, 0.0)
    g_a1, w_a1 = run(int8, True, 1.0)
    tag = "int8" if int8 else "bf16"
    assert torch.equal(w_off, w_a0) and torch.equal(w_off, w_a1), f"{tag}: flips changed"
    assert torch.equal(g_off, g_a0), f"{tag}: alpha=0 is not identical to EF off"
    assert torch.isfinite(g_a1).all(), f"{tag}: non-finite grad_x"
    rel = ((g_a1 - g_off).norm() / g_off.norm()).item()
    cos = ((g_a1 * g_off).sum() / (g_a1.norm() * g_off.norm())).item()
    assert 0.05 < rel < 5.0, f"{tag}: feedback term has implausible size {rel}"
    flips = (w_off != run(int8, False, 0.0, seed=1000)[1]).sum().item()
    print(f"{tag}: alpha=0 identical to off | flips unchanged by EF | "
          f"alpha=1: |dgx|/|gx| = {rel:.3f}, cos(gx_ef, gx) = {cos:.3f}")

# guards
c = ModelConfig(vocab_size=1000, dim=64, n_layers=1, n_heads=4, n_kv_heads=4,
                hidden_dim=128, max_seq_len=32)
m = build_kernel_transformer(c, evidence=True, ev_bits=3)
try:
    set_err_feedback(m, True); raise AssertionError("evidence mode not rejected")
except ValueError:
    pass
m = build_kernel_transformer(c)
for l in m.modules():
    if isinstance(l, KernelTernaryLinear):
        l.accum_steps = 4
try:
    set_err_feedback(m, True); raise AssertionError("grad accumulation not rejected")
except ValueError:
    pass
print("guards: evidence mode and grad_accum>1 rejected")
print("OK: spatial error feedback")
