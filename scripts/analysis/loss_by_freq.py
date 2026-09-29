"""Where is master's lead, by how often the model has seen the thing it predicts? Validation loss (the same 48
sequences as loss_by_pos) split by the training-set count of the target token, and of the (previous, target) pair.
A lead on rare pairs points at integration over a long horizon (a signal present in a few batches only); a lead
that is the same at every count points elsewhere. For each checkpoint given as RUN:STEP:KIND.
  python -m scripts.analysis.loss_by_freq curve_master:3000:master gvsharp_rc_s0:3000:kernel ...
"""
import sys, numpy as np, torch, torch.nn.functional as F
from scripts.analysis.induction_heads import load

val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(7)
starts = g.integers(0, len(val) - 2049, 48)
X = torch.tensor(np.stack([val[s:s + 2049].astype(np.int64) for s in starts]))
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
uni = np.bincount(np.asarray(tr), minlength=65536)
prev, tgt = X[:, 1:-1].numpy().ravel(), X[:, 2:].numpy().ravel()          # pairs from position 2 on
pid = prev.astype(np.int64) * 65536 + tgt
up, inv = np.unique(pid, return_inverse=True)
cnt = np.zeros(len(up), np.int64)
for a in range(0, len(tr) - 1, 50_000_000):                               # count only the pairs val uses
    c = np.asarray(tr[a:a + 50_000_001]).astype(np.int64)
    q = c[:-1] * 65536 + c[1:]
    i = np.searchsorted(up, q); i[i >= len(up)] = 0
    hit = up[i] == q
    cnt += np.bincount(i[hit], minlength=len(up))
pc = cnt[inv].reshape(48, -1)
tc = uni[X[:, 2:].numpy()]
UB = [(0, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e12)]
PB = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 1e4), (1e4, 1e12)]
share = lambda c, bs: "  ".join(f"{100 * np.mean((c >= a) & (c < b)):5.1f}" for a, b in bs)
print("share of val positions (%)  token count <1e4 1e4-1e5 1e5-1e6 >1e6 :", share(tc, UB))
print("share of val positions (%)  pair count 0 1-9 10-99 1e2-1e3 1e3-1e4 >1e4 :", share(pc, PB))
print("run @step".ljust(26) + "| token count: <1e4  1e4-5  1e5-6  >1e6 | pair count:   0    1-9  10-99  1e2-3  1e3-4  >1e4 | all")
for arg in sys.argv[1:]:
    run, st, kind = arg.split(":")
    step, m, V = load(f"checkpoints/{run}/ckpt_{st}.pt", kind)
    L = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(X)):
            x = X[i:i + 1, :-1].cuda(); y = X[i:i + 1, 1:].cuda()
            out = m(x); lg = out[0] if isinstance(out, tuple) else out
            L.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).cpu())
    L = torch.cat(L)[:, 1:].numpy()                                        # targets at positions 2..2048
    f = lambda c, bs: "  ".join(f"{L[(c >= a) & (c < b)].mean():5.2f}" for a, b in bs)
    print(f"{run} @{st}".ljust(26) + f"|             {f(tc, UB)} |            {f(pc, PB)} | {L.mean():.3f}", flush=True)
    del m; torch.cuda.empty_cache()
