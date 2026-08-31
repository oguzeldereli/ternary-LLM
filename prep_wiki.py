"""Stream English Wikipedia -> GPT-2 BPE uint16 token stream (~1.25B tokens).

Writes data/wiki_all.bin incrementally (low RAM), then splits off a val tail.
<eot> (50256) separates articles.
"""
import sys, time, numpy as np, tiktoken
from datasets import load_dataset

TARGET = int(float(sys.argv[1]) if len(sys.argv) > 1 else 1.25e9)  # tokens
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                  streaming=True)

n = 0; ndoc = 0; t0 = time.time(); nextlog = 50_000_000
with open("data/wiki_all.bin", "wb") as f:
    for ex in ds:
        ids = enc.encode_ordinary(ex["text"])
        ids.append(EOT)
        np.array(ids, dtype=np.uint16).tofile(f)
        n += len(ids); ndoc += 1
        if n >= nextlog:
            dt = time.time() - t0
            print(f"{n/1e6:.0f}M tokens | {ndoc:,} docs | {n/dt/1e3:.0f}k tok/s "
                  f"| {dt/60:.1f} min", flush=True)
            nextlog += 50_000_000
        if n >= TARGET:
            break

print(f"done streaming: {n:,} tokens, {ndoc:,} docs, {(time.time()-t0)/60:.1f} min",
      flush=True)

# split off val tail
d = np.memmap("data/wiki_all.bin", dtype=np.uint16, mode="r")
nval = 2_000_000
d[:-nval].tofile("data/wiki_train.bin")
d[-nval:].tofile("data/wiki_val.bin")
print(f"train {len(d)-nval:,} | val {nval:,} -> data/wiki_train.bin, data/wiki_val.bin",
      flush=True)
