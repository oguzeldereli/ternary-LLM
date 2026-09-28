"""Why did --lr_refresh hurt? On the step-4000 bench, per layer: pick the 16 directions of M's rank-256 row space
that catch the least gradient (judged on the 3 batch gradients of the bench, a stand-in for the run's ~10-step
EMA), and the 16 replacement directions (top directions of one batch's gradient outside the subspace). Compare:
  - M's own energy in the dropped directions (momentum thrown away)
  - true-gradient energy per direction: dropped vs kept vs new vs random directions outside the subspace
  - whether "weak on batches" means "weak on the true gradient" (rank correlation)

  python -m scripts.analysis.refresh_check
"""
import json, torch, numpy as np
from scripts.analysis.testbench import OUT
dev = "cuda"
meta = json.load(open(f"{OUT}/meta.json")); names = meta["ternary_layers"]
b = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
ld = lambda f: torch.load(f"{OUT}/{f}.pt")
G, g0, g1, g2 = ld("gbar"), ld("g_step"), ld("g_la1"), ld("g_la2")
k = 16; out = []
torch.manual_seed(0)
for i, (U, V) in enumerate(b["lowrank"]):
    U, V = U.to(dev).float(), V.to(dev).float()
    gb = G[names[i]].to(dev); gs = [x[names[i]].to(dev) for x in (g0, g1, g2)]
    e_batch = sum((g @ V).pow(2).sum(0) / g.pow(2).sum() for g in gs) / 3        # per-direction share, batches
    e_true = (gb @ V).pow(2).sum(0) / gb.pow(2).sum()                             # ... on the true gradient
    drop = e_batch.argsort()[:k]; keep = e_batch.argsort()[k:]
    m_energy = U.pow(2).sum(0); m_drop = (m_energy[drop].sum() / m_energy.sum()).item()
    # replacement directions, as in the trainer: top residual directions of one batch gradient
    g = gs[0]; Vk = V[:, keep]
    R = g - (g @ Vk) @ Vk.T
    Q = torch.linalg.qr(R.T @ (R @ torch.randn(R.shape[1], k + 4, device=dev)))[0][:, :k]
    Q = torch.linalg.qr(Q - Vk @ (Vk.T @ Q))[0]
    e_new = ((gb @ Q).pow(2).sum(0) / gb.pow(2).sum()).mean().item()
    Z = torch.randn(V.shape[0], k, device=dev); Z = torch.linalg.qr(Z - V @ (V.T @ Z))[0]
    e_rand = ((gb @ Z).pow(2).sum(0) / gb.pow(2).sum()).mean().item()
    rb = e_batch.argsort().argsort().float(); rt = e_true.argsort().argsort().float()
    spear = torch.corrcoef(torch.stack([rb, rt]))[0, 1].item()
    out.append([m_drop, e_true[drop].mean().item(), e_true[keep].mean().item(), e_new, e_rand, spear,
                (e_true[drop].sum() / e_true.sum()).item()])
R = np.array(out)
print(f"layers {len(R)}; per refresh: 16 of 256 directions")
print(f"M's own energy in the 16 dropped directions (share of M):            {R[:, 0].mean():.3f}")
print(f"true-gradient share per direction: dropped {R[:, 1].mean():.2e} | kept {R[:, 2].mean():.2e} | "
      f"new {R[:, 3].mean():.2e} | random outside {R[:, 4].mean():.2e}")
print(f"share of the subspace's true-gradient energy lost with the dropped: {R[:, 6].mean():.3f}")
print(f"rank correlation, batch-judged vs true per-direction energy:          {R[:, 5].mean():.3f}")
