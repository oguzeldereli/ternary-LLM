"""Cleaned proposal signals on the step-4000 bench: which transform of M (and the batch gradients available at
that step) proposes flips that go with the true gradient? For each signal: proposals by the flip rule at the
run's rate, then the trainer's look-ahead test (x2, the bench's saved look-ahead gradients are at the M
proposal point, so for other signals the test uses g_step + g_la1 + g_la2 as fixed votes). Scores against gbar.

  python -m scripts.analysis.bench_signals
"""
import json, torch
from scripts.analysis.testbench import Bench, score, OUT, dev

meta = json.load(open(f"{OUT}/meta.json"))
B = Bench(meta["ckpt"]); rate = meta["rate"]
ld = lambda f: (lambda G: [G[n].to(dev) for n in B.names])(torch.load(f"{OUT}/{f}.pt"))
gbar, g, g1, g2 = ld("gbar"), ld("g_step"), ld("g_la1"), ld("g_la2")
M = [(U.to(dev).float() @ V.to(dev).float().T) for U, V in B.b["lowrank"]]
B.L0 = B.held_out()
unit = lambda X: [x / x.abs().mean().clamp_min(1e-30) for x in X]     # per-layer scale-free
Mu, gu = unit(M), unit(g)


def rcnorm(X):
    out = []
    for x in X:
        r = x.pow(2).mean(1, keepdim=True).sqrt(); c = x.pow(2).mean(0, keepdim=True).sqrt()
        out.append(x / (r * c).sqrt().clamp_min(1e-30))
    return out


signals = {
    "M (current)": M,
    "g (one batch)": g,
    "M gated by sign(g)": [m * (m.sign() == a.sign()) for m, a in zip(M, g)],
    "M gated by 2 of 3 batches": [m * (((m.sign() == a.sign()).int() + (m.sign() == b.sign()).int()
                                        + (m.sign() == c.sign()).int()) >= 2) for m, a, b, c in zip(M, g, g1, g2)],
    "M + 1.0 g (unit-scaled)": [a + b for a, b in zip(Mu, gu)],
    "M + 0.3 g (unit-scaled)": [a + 0.3 * b for a, b in zip(Mu, gu)],
    "row/col-normalized M": rcnorm(M),
    "row/col-normalized g": rcnorm(g),
}
print(f"{'signal':32s} {'':9s} {'flips':>8s} {'precision':>9s} {'D.gbar':>10s} {'cos':>8s} {'held-out dL':>11s}")
res = {}
for name, S in signals.items():
    D = B.propose(S, rate)
    rp = score(B, D, gbar)
    K = [d * ((d.float() * (a + b)) < 0).to(torch.int8) for d, a, b in zip(D, g, g1)]
    K = [d * ((d.float() * (a + b)) < 0).to(torch.int8) for d, a, b in zip(K, g, g2)]
    rk = score(B, K, gbar)
    res[name] = {"proposed": rp, "kept": rk}
    for tag, r in (("proposed", rp), ("kept", rk)):
        print(f"{name if tag == 'proposed' else '':32s} {tag:9s} {r['flips']:8d} {r['precision']:9.3f} "
              f"{-r['gain']:+10.3e} {r['cos']:+8.4f} {r['held_out_dL']:+11.4f}", flush=True)
json.dump(res, open(f"{OUT}/signals.json", "w"), indent=1)
