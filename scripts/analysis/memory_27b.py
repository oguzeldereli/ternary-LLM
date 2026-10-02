"""Memory of a ~23B-parameter ternary model (width 5120, 83 layers, MLP 13824, 8 KV heads of 128) trained with the
two-timescale rule, for different ranks and storage formats of the state. Estimate: weights, float tail (16-bit values,
gradients and 8-bit Adam, ~1.6 GiB), the gradient of the largest layer (update inside the backward) and the state;
activations are extra (~3 GiB at micro-batch 1 x 2048 with full checkpointing).
    python scripts/analysis/memory_27b.py"""
GiB = 2**30
d, L, f, kv = 5120, 83, 13824, 1024
mats = [(d, d), (kv, d), (kv, d), (d, d), (f, d), (f, d), (d, f)]  # (out, in): q k v o gate up down
tern = L * sum(n * k for n, k in mats); P = tern + 0.29e9
nk = L * sum(n + k for n, k in mats); Nsum = L * sum(n for n, _ in mats); Ksum = L * sum(k for _, k in mats)
packed = tern / 5; tail = 1.6 * GiB; grad = max(n * k for n, k in mats) * 4


def row(name, r_s, r, bm, bU, bV, host=False):
    st = nk * r_s * bm + (Nsum * bU + Ksum * bV) * r + nk * 4  # momentum, accumulator U / V, factored second moment
    tot = packed + tail + grad + st
    gpu = tot - st + (2 * st / L if host else st)                # host: two layers' state on the GPU at a time
    print(f"{name:46s} state {st / GiB:5.2f} GiB | GPU {gpu / GiB:5.2f} GiB ({gpu / 1e9:4.1f} GB) + activations"
          f" | {tot / P:.2f} B/param ({16 / (tot / P):4.1f}x less than 16)")


print(f"ternary {tern / 1e9:.2f}B, all {P / 1e9:.2f}B params; packed trits {packed / GiB:.2f} GiB; largest gradient {grad / GiB:.2f} GiB")
row("all 8-bit, ranks 1024 (earlier estimate)", 1024, 1024, 1, 1, 1)
row("8-bit direction + 16-bit accumulator, 1024", 1024, 1024, 1, 2, 2)
row("  same, accumulator V in 8-bit (untested)", 1024, 1024, 1, 2, 1)
row("8-bit / 16-bit, ranks 512", 512, 512, 1, 2, 2)
row("8-bit / 16-bit, ranks 256", 256, 256, 1, 2, 2)
row("8-bit / 16-bit, ranks 1024, state in host RAM", 1024, 1024, 1, 2, 2, host=True)
row("fp32 state, ranks 1024", 1024, 1024, 4, 4, 4)
