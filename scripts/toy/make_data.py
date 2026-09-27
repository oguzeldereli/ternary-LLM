"""Synthetic induction data for the toy2 preset: a stream of random segments (tokens uniform over the vocab,
length uniform in [8, 64]), each immediately repeated once. The first copy is unpredictable (loss ~ log V);
the repeat is predictable only by in-context copying in order (an induction head).

  python -m scripts.toy.make_data      -> data/toy_ind_train.bin (20M tokens), data/toy_ind_val.bin (1M)
"""
import os, numpy as np

V = int(os.environ.get("TOY_V", 2048)); LO, HI = 8, 64


def stream(n, seed):
    g = np.random.default_rng(seed); out = []; tot = 0
    while tot < n:
        seg = g.integers(0, V, g.integers(LO, HI + 1))
        out += [seg, seg]; tot += 2 * len(seg)
    return np.concatenate(out)[:n].astype(np.uint16)


if __name__ == "__main__":
    tag = "" if V == 2048 else f"_v{V}"                  # TOY_V=256 -> data/toy_ind_v256_{train,val}.bin
    stream(20_000_000, 1).tofile(f"data/toy_ind{tag}_train.bin")
    stream(1_000_000, 2).tofile(f"data/toy_ind{tag}_val.bin")
    print(f"wrote data/toy_ind{tag}_train.bin, data/toy_ind{tag}_val.bin")
