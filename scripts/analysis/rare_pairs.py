"""Why is the gap to master largest for (previous, target) pairs seen 10-999 times in training? Held-out loss (the same
48 sequences as loss_by_freq) in finer pair-count buckets, split by the target token's kind (word start "▁..." vs word
continuation), with each run's share of master's gain over the unigram guess: (U - ours) / (U - master), U = the
unigram cross-entropy of the target (-log of its training-set frequency).
  python -m scripts.analysis.rare_pairs master_tracked:9154:master gvsharp_dryspend_r1024_s0:9154:kernel ...
"""
import sys, numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from scripts.analysis.induction_heads import load

val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(7)
starts = g.integers(0, len(val) - 2049, 48)
X = torch.tensor(np.stack([val[s:s + 2049].astype(np.int64) for s in starts]))
tr = np.memmap("data/wiki32k_train.bin", dtype=np.uint16, mode="r")
uni = np.bincount(np.asarray(tr), minlength=65536).astype(np.float64)
prev, tgt = X[:, 1:-1].numpy().ravel(), X[:, 2:].numpy().ravel()
pid = prev.astype(np.int64) * 65536 + tgt
up, inv = np.unique(pid, return_inverse=True)
cnt = np.zeros(len(up), np.int64)
for a in range(0, len(tr) - 1, 50_000_000):
    c = np.asarray(tr[a:a + 50_000_001]).astype(np.int64); q = c[:-1] * 65536 + c[1:]
    i = np.searchsorted(up, q); i[i >= len(up)] = 0; hit = up[i] == q
    cnt += np.bincount(i[hit], minlength=len(up))
PC = cnt[inv]
U = -np.log(uni[tgt] / uni.sum() + 1e-12)                                    # unigram guess
tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
start = np.array([tok.id_to_token(int(i)).startswith("▁") for i in range(32000)])
KIND = start[tgt]                                                           # True = the target starts a word
EDGES = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1 << 62]
LAB = ["0", "1-2", "3-9", "10-29", "30-99", "100-299", "300-999", "1k-3k", "3k-10k", ">10k"]
B = np.digitize(PC, EDGES[1:-1])
print("bucket   " + "  ".join(f"{l:>8s}" for l in LAB))
print("share %  " + "  ".join(f"{100 * np.mean(B == k):8.1f}" for k in range(len(LAB))))
print("start %  " + "  ".join(f"{100 * KIND[B == k].mean():8.1f}" for k in range(len(LAB))), " (share of targets that start a word)")
print("unigram  " + "  ".join(f"{U[B == k].mean():8.2f}" for k in range(len(LAB))))
res = {}
for arg in sys.argv[1:]:
    run, st, kind = arg.split(":")
    step, m, V = load(f"checkpoints/{run}/ckpt_{st}.pt", kind)
    L = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(X)):
            x = X[i:i + 1, :-1].cuda(); y = X[i:i + 1, 1:].cuda()
            out = m(x); lg = out[0] if isinstance(out, tuple) else out
            L.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).cpu())
    res[f"{run}@{st}"] = torch.cat(L)[:, 1:].numpy().ravel()
    del m; torch.cuda.empty_cache()
keys = list(res); M = res[keys[0]]
for k in keys:
    L = res[k]
    print(f"\n{k}  (all {L.mean():.3f})")
    for nm, sel in (("all      ", np.ones_like(KIND)), ("wordstart", KIND), ("continue ", ~KIND)):
        row = []
        for b in range(len(LAB)):
            s = sel & (B == b)
            row.append(f"{L[s].mean():8.2f}" if k == keys[0] else f"{L[s].mean() - M[s].mean():+8.3f}")
        print(f"  {nm}" + "  ".join(row) + ("  (loss)" if k == keys[0] else "  (minus master)"))
    if k != keys[0]:
        fr = [(U[B == b] - L[B == b]).mean() / max((U[B == b] - M[B == b]).mean(), 1e-9) for b in range(len(LAB))]
        print("  of master's gain over unigram: " + "  ".join(f"{100 * f:6.1f}%" for f in fr))
