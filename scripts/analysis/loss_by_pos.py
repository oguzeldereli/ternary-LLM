"""Which part of the loss has a run learned? Validation loss split by position in the context (0-1: no context to
use; 2-15: local; 16-127; 128-511; 512+: long context through attention), plus the ordered-copy (induction) gain on
random tokens. For each checkpoint given as RUN:STEP:KIND.
  python -m scripts.analysis.loss_by_pos rc_s0:1000:kernel user_cap1_s0:1000:kernel ...
"""
import sys, numpy as np, torch, torch.nn.functional as F
from scripts.analysis.induction_heads import load

val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(7)
starts = g.integers(0, len(val) - 2049, 48)
X = torch.tensor(np.stack([val[s:s + 2049].astype(np.int64) for s in starts]))
B = [(0, 2), (2, 16), (16, 128), (128, 512), (512, 2048)]
print("run @step (tokens)            pos 0-1  2-15  16-127  128-511  512+   all   copy gain (exact - shuffled)")
for arg in sys.argv[1:]:
    run, st, kind = arg.split(":")
    step, m, V = load(f"checkpoints/{run}/ckpt_{st}.pt", kind)
    L = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 8):
            x = X[i:i + 8, :-1].cuda(); y = X[i:i + 8, 1:].cuda()
            out = m(x); lg = out[0] if isinstance(out, tuple) else out
            L.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).cpu())
        L = torch.cat(L)
        gains = []
        for shuf in (False, True):
            gg = torch.Generator().manual_seed(99); gs = []
            for _ in range(8):
                r = torch.randint(1000, V - 1000, (16, 64), generator=gg).cuda()
                r2 = r[:, torch.randperm(64, generator=gg).cuda()] if shuf else r
                x = torch.cat([r, r2], 1); out = m(x[:, :-1]); lg = out[0] if isinstance(out, tuple) else out
                l = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(16, -1)
                gs.append(float(l[:, 1:63].mean() - l[:, 65:].mean()))
            gains.append(np.mean(gs))
    cols = "  ".join(f"{float(L[:, a:b].mean()):6.3f}" for a, b in B)
    print(f"{run} @{st} ({int(st) * 32768 / 1e6:.0f}M)".ljust(30) + f"{cols}  {float(L.mean()):6.3f}   {gains[0] - gains[1]:+.3f}", flush=True)
    del m; torch.cuda.empty_cache()
