"""Stream FineWeb-Edu -> uint16 token streams (Llama-2 32k tokenizer, the same as wiki32k).

  python3 -m scripts.data.prep_fineweb 2e9 --out_prefix data/fwedu32k

The first --val_tokens tokens' worth of documents go to <prefix>_val.bin (held out, written first, so no copy of the
whole stream is needed), the rest to <prefix>_train.bin until the target. An end-of-text token (</s>, 2) separates
documents. Needs: pip install datasets tokenizers.
"""
import argparse, time, numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("target", type=float, nargs="?", default=2e9, help="training tokens")
ap.add_argument("--name", default="sample-10BT", help="FineWeb-Edu subset")
ap.add_argument("--out_prefix", default="data/fwedu32k")
ap.add_argument("--val_tokens", type=int, default=10_000_000)
ap.add_argument("--batch", type=int, default=1024, help="documents per encode batch")
args = ap.parse_args()
TARGET = int(args.target)

tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
EOT = tok.token_to_id("</s>")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.name, split="train", streaming=True)

nval = ntr = ndoc = 0; t0 = time.time(); nextlog = 100_000_000; buf = []
fv = open(args.out_prefix + "_val.bin", "wb"); ft = open(args.out_prefix + "_train.bin", "wb")


def flush(texts):
    global nval, ntr, ndoc
    for e in tok.encode_batch(texts, add_special_tokens=False):
        ids = np.array(e.ids + [EOT], dtype=np.uint16)
        if nval < args.val_tokens:
            ids.tofile(fv); nval += len(ids)
        else:
            ids.tofile(ft); ntr += len(ids)
        ndoc += 1


for ex in ds:
    buf.append(ex["text"])
    if len(buf) == args.batch:
        flush(buf); buf = []
        if ntr >= nextlog:
            dt = time.time() - t0
            print(f"{ntr / 1e6:.0f}M train tokens | {ndoc:,} docs | {(ntr + nval) / dt / 1e3:.0f}k tok/s | "
                  f"{dt / 60:.1f} min", flush=True)
            nextlog += 100_000_000
        if ntr >= TARGET:
            break
if buf and ntr < TARGET:
    flush(buf)
fv.close(); ft.close()
print(f"done: train {ntr:,} tokens, val {nval:,} tokens, {ndoc:,} docs, {(time.time() - t0) / 60:.1f} min -> "
      f"{args.out_prefix}_train.bin, {args.out_prefix}_val.bin", flush=True)
