"""Measure evidence density from the live 3-bit checkpoint, then test whether a
d=2^20 superposition (Count-Sketch) can recover it 'good enough' for flip gating."""
import torch

blob = torch.load("checkpoints/wiki_ev3/ckpt.pt", map_location="cpu", weights_only=False)
sd = blob["model"]
evs = [v.flatten() for k, v in sd.items() if k.endswith(".evidence")]
e = torch.cat(evs).to(torch.float32)          # {-3..3}, one per ternary weight
N = e.numel()
nz = e != 0
K = int(nz.sum())
print(f"step {blob.get('step')} | evidence entries N = {N:,}")
print(f"DENSITY (nonzero): {K/N*100:.2f}%   (K = {K:,} active)")
vals, cnts = e.unique(return_counts=True)
print("value histogram:", {int(v): int(c) for v, c in zip(vals, cnts)})

d = 2 ** 20
print(f"\nsuperposition d = 2^20 = {d:,} | load K/d = {K/d:.2f} "
      f"(theory SNR ~ sqrt(d/K) = {(d/max(K,1))**0.5:.2f})")

def count_sketch(e, d, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    ests = []
    for _ in range(t):
        h = torch.randint(0, d, (N,), generator=g)
        s = (torch.randint(0, 2, (N,), generator=g) * 2 - 1).to(torch.float32)
        M = torch.zeros(d)
        M.index_add_(0, h, s * e)              # superpose
        ests.append(s * M[h])                  # unbind (dot with own key)
    est = torch.stack(ests).median(0).values if t > 1 else ests[0]
    return est

for t in (1, 3):
    est = count_sketch(e, d, t)
    # metrics on the entries that matter (nonzero evidence -> gate a flip)
    sign_acc = (torch.sign(est[nz]) == torch.sign(e[nz])).float().mean().item()
    # also: does it wrongly wake zeros (false evidence on zero weights)?
    zero = ~nz
    false_fire = (est[zero].abs() >= 1.0).float().mean().item()
    corr = torch.corrcoef(torch.stack([est[nz], e[nz]]))[0, 1].item()
    print(f"  t={t} tables: sign-recovery on nonzeros {sign_acc*100:.1f}% | "
          f"corr {corr:.3f} | false-eco on zeros >=1: {false_fire*100:.1f}%")

# reference: what does a plain 3-bit array cost vs this sketch?
print(f"\nmemory: dense 3-bit array = {N*3/8/1e6:.1f} MB | "
      f"sketch d cells @ int8 = {d/1e6:.1f} MB (holds all N IF recoverable)")
