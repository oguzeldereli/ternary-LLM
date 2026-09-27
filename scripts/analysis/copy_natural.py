"""Ordered copying on natural text vs random tokens: a 64-token segment followed by its exact repeat or a shuffled
repeat; gain = loss on the first copy minus loss on the repeat (tokens 2..64). Random: tokens uniform in
[1000, V-1000] (the induction tracker's test). Natural: real validation segments.
  python -m scripts.analysis.copy_natural"""
import os, numpy as np, torch, torch.nn.functional as F
from scripts.analysis.induction_heads import load

RUNS = [("master, 164M", "checkpoints/curve_master/ckpt_5000.pt", "master", ""),
        ("no LA then LA at 131M, 164M", "checkpoints/nola_then_la/ckpt_5000.pt", "kernel", ""),
        ("look-ahead + additive r4, 164M", "checkpoints/magadd_full/ckpt_5000.pt", "kernel", "add"),
        ("look-ahead + additive r4, 295M", "checkpoints/magadd_full/ckpt_9000.pt", "kernel", "add"),
        ("look-ahead baseline, 300M", "checkpoints/lm_lowrank256_xb2_100M/ckpt.pt", "kernel", ""),
        ("no look-ahead, 295M", "checkpoints/nola_lab/ckpt_9000.pt", "kernel", "")]
val = np.memmap("data/wiki32k_val.bin", dtype=np.uint16, mode="r")
g = np.random.default_rng(5)
nat = torch.from_numpy(np.stack([val[i:i + 64].astype(np.int64) for i in g.integers(0, len(val) - 64, 128)])).cuda()


def gains(m, seg):
    out = []
    for shuf in (False, True):
        gen = torch.Generator().manual_seed(1)
        r2 = seg[:, torch.randperm(64, generator=gen).cuda()] if shuf else seg
        x = torch.cat([seg, r2], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lp = torch.cat([F.cross_entropy(m(x[i:i + 16, :-1])[0].float().transpose(1, 2), x[i:i + 16, 1:],
                                            reduction="none") for i in range(0, x.shape[0], 16)])
        out.append((lp[:, :63].mean() - lp[:, 64:].mean()).item())
    return out


print(f"{'model':32s} {'random: exact':>13s} {'shuffled':>9s} {'ordered':>8s} | {'natural: exact':>14s} {'shuffled':>9s} {'ordered':>8s}")
for name, path, kind, mag in RUNS:
    os.environ["MAG_KIND"] = mag or "add"
    s, m, V = load(path, kind)
    rnd = torch.from_numpy(np.random.default_rng(7).integers(1000, V - 1000, (128, 64))).cuda()
    re_, rs = gains(m, rnd); ne, ns = gains(m, nat)
    print(f"{name:32s} {re_:13.3f} {rs:9.3f} {re_ - rs:8.3f} | {ne:14.3f} {ns:9.3f} {ne - ns:8.3f}", flush=True)
    del m; torch.cuda.empty_cache()
