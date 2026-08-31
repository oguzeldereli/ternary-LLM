"""Why ε doesn't save it: superposition crosstalk = sqrt(K)*rms_coherence =
sqrt(K/d), independent of the *pairwise* coherence. Sweep d on real evidence."""
import math, torch

blob = torch.load("checkpoints/wiki_ev3/ckpt.pt", map_location="cpu", weights_only=False)
sd = blob["model"]
e = torch.cat([v.flatten() for k, v in sd.items() if k.endswith(".evidence")]).float()
N = e.numel(); nz = e != 0; K = int(nz.sum())
print(f"N={N:,}  K(active)={K:,}  density={K/N*100:.1f}%")

def sketch(d, t=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    ests = []
    for _ in range(t):
        h = torch.randint(0, d, (N,), generator=g)
        s = (torch.randint(0, 2, (N,), generator=g) * 2 - 1).float()
        M = torch.zeros(d); M.index_add_(0, h, s * e)
        ests.append(s * M[h])
    return torch.stack(ests).median(0).values if t > 1 else ests[0]

print(f"\n{'d':>12} {'ε≈1/√d':>8} {'load K/d':>9} {'sign-rec':>9} {'corr':>6} {'sketch MB':>10}")
for p in (20, 22, 24, 25, 26):
    d = 2 ** p
    est = sketch(d)
    sa = (torch.sign(est[nz]) == torch.sign(e[nz])).float().mean().item()
    cc = torch.corrcoef(torch.stack([est[nz], e[nz]]))[0, 1].item()
    eps = 1 / math.sqrt(d)
    print(f"{d:>12,} {eps:>8.4f} {K/d:>9.2f} {sa*100:>8.1f}% {cc:>6.2f} {d/1e6:>9.1f}M")

print(f"\ndense 3-bit array (exact): {N*3/8/1e6:.1f} MB")
print(f"d needed for load<=1 (SNR~1): d>=K={K:,} = {K/1e6:.0f}M cells "
      f"-> {K/1e6:.0f} MB int8, already > the 9.6MB exact array")
