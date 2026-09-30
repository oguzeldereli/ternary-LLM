"""Loss by context position with error bars: NSEQ held-out sequences (default 192, 4x loss_by_pos), per position bucket
the mean loss of each run minus the first run's (master), with a bootstrap standard error over sequences (paired: the
same sequences for every run).
  python -m scripts.analysis.pos_gap master_tracked:9154:master gvsharp_dry_s0:9154:kernel ... [--n 192]
"""
import sys, numpy as np, torch, torch.nn.functional as F
from scripts.analysis.induction_heads import load

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NSEQ = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 192
args = [a for a in args if not a.isdigit()]
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
starts = np.random.default_rng(21).integers(0, len(val) - 2049, NSEQ)
X = torch.tensor(np.stack([val[s:s + 2049].astype(np.int64) for s in starts]))
B = [(0, 2), (2, 16), (16, 128), (128, 512), (512, 2048)]
per = {}
for arg in args:
    run, st, kind = arg.split(":")
    step, m, V = load(f"checkpoints/{run}/ckpt_{st}.pt", kind)
    L = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(NSEQ):
            x = X[i:i + 1, :-1].cuda(); y = X[i:i + 1, 1:].cuda()
            out = m(x); lg = out[0] if isinstance(out, tuple) else out
            L.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").cpu())
    per[run] = torch.stack(L).numpy()                 # NSEQ x 2048
    del m; torch.cuda.empty_cache()
rng = np.random.default_rng(0)
boot = rng.integers(0, NSEQ, (2000, NSEQ))
names = list(per); M = per[names[0]]
print(f"{NSEQ} sequences; tokens per bucket: " + " / ".join(str(NSEQ * (b - a)) for a, b in B))
print(f"{'run':30s}" + "".join(f"{f'{a}-{b - 1}':>18s}" for a, b in B))
for n in names:
    cells = []
    for a, b in B:
        d = (per[n][:, a:b] - M[:, a:b]).mean(1)       # per-sequence mean difference
        if n == names[0]:
            cells.append(f"{per[n][:, a:b].mean():18.3f}")
        else:
            se = d[boot].mean(1).std()
            cells.append(f"{d.mean():+10.3f} ± {se:.3f}")
    print(f"{n:30s}" + "".join(cells))
