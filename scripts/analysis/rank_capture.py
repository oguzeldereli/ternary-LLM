"""Does the rank-256 momentum even hold the gradient that teaches rare pairs? At one checkpoint with its saved
momentum M = U V^T: the held-out gradient restricted to target positions by the training-set count of their
(previous, target) pair, and the share of that gradient's signal inside M's row x column subspace. Each bucket's
gradient is measured on two disjoint halves of the sequences; signal energy = <G1, G2> (noise cancels), captured =
<P G1, P G2> with P the projection onto span(U) x span(V). Also: the best any rank-256 subspace could do (the top-256
SVD of G1 + G2, checked on the cross product) and cos(M, G) with G = G1 + G2.
  python -m scripts.analysis.rank_capture RUN STEP [NSEQ]
"""
import sys, numpy as np, torch
from scripts.analysis.testbench import Bench

RUN, ST = sys.argv[1], int(sys.argv[2]); NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 256
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
starts = np.random.default_rng(11).integers(0, len(val) - 2049, NSEQ)
X = np.stack([val[s:s + 2049].astype(np.int64) for s in starts])
pid = (X[:, :-1] * 65536 + X[:, 1:]).ravel()                         # (input token, target) pairs
up, inv = np.unique(pid, return_inverse=True); cnt = np.zeros(len(up), np.int64)
for a in range(0, len(tr) - 1, 50_000_000):
    c = np.asarray(tr[a:a + 50_000_001]).astype(np.int64); q = c[:-1] * 65536 + c[1:]
    i = np.searchsorted(up, q); i[i >= len(up)] = 0; hit = up[i] == q
    cnt += np.bincount(i[hit], minlength=len(up))
PC = cnt[inv].reshape(NSEQ, 2048)
BK = {"pairs seen 0-999x": (0, 1000), "1e3-1e4x": (1000, 10000), ">1e4x": (10000, 1 << 62)}

B = Bench(f"checkpoints/{RUN}/ckpt_{ST}.pt")
Ls, names = B.Ls, B.names
UV = [(U.cuda().float(), V.cuda().float()) for U, V in B.b["lowrank"]]
Q = [(torch.linalg.qr(U)[0], torch.linalg.qr(V)[0]) for U, V in UV]
Xt = torch.tensor(X)


def bucket_grad(lo, hi, half):
    acc, n = None, 0
    idx = np.arange(half, NSEQ, 2)
    for j in range(0, len(idx), 8):
        s = idx[j:j + 8]
        x = Xt[s, :-1].cuda(); y = Xt[s, 1:].clone()
        keep = torch.tensor((PC[s] >= lo) & (PC[s] < hi)); y[~keep] = -1
        k = int(keep.sum())
        if k == 0: continue
        gd = B.grad([(x, y.cuda())])
        g = [gd[nm].float() * k for nm in names]                          # the loss is a mean over kept targets
        acc = g if acc is None else [a + b for a, b in zip(acc, g)]; n += k
    return [a / n for a in acc], n


print(f"{RUN} @{ST}: {NSEQ} held-out sequences; share of positions per bucket:",
      "  ".join(f"{k} {100 * np.mean((PC >= a) & (PC < b)):.1f}%" for k, (a, b) in BK.items()))
print(f"{'bucket':18s} {'signal/noise':>12s} {'in M subspace':>14s} {'rows only':>10s} {'cols only':>10s} "
      f"{'best rank-256':>14s} {'random 256':>11s} {'cos(M,G)':>9s}")
for name, (lo, hi) in BK.items():
    G1, n1 = bucket_grad(lo, hi, 0); G2, n2 = bucket_grad(lo, hi, 1)
    sig = cap = rows = cols = best = rnd = 0.0; noise = 0.0; mg = mm = gg = 0.0
    for (Qu, Qv), (U, V), a, b in zip(Q, UV, G1, G2):
        sig += float((a * b).sum()); noise += float(((a - b) ** 2).sum()) / 4
        pa, pb = Qu.T @ a @ Qv, Qu.T @ b @ Qv; cap += float((pa * pb).sum())
        rows += float(((Qu.T @ a) * (Qu.T @ b)).sum()); cols += float(((a @ Qv) * (b @ Qv)).sum())
        Us, _, Vs = torch.linalg.svd(a + b, full_matrices=False); Us, Vs = Us[:, :256], Vs[:256].T
        best += float(((Us.T @ a @ Vs) * (Us.T @ b @ Vs)).sum())
        rnd += float((a * b).sum()) * (256 / a.shape[0]) * (256 / a.shape[1])
        M = U @ V.T; G = a + b
        mg += float((M * G).sum()); mm += float((M * M).sum()); gg += float((G * G).sum())
    print(f"{name:18s} {sig / max(noise, 1e-30):12.2f} {100 * cap / sig:13.1f}% {100 * rows / sig:9.1f}% "
          f"{100 * cols / sig:9.1f}% {100 * best / sig:13.1f}% {100 * min(rnd / sig, 1):10.1f}% {mg / (mm * gg) ** .5:+9.3f}",
          flush=True)
