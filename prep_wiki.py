"""Stream English Wikipedia -> uint16 token stream.

  python3 prep_wiki.py 1.25e9                                   # GPT-2 BPE -> data/wiki_*
  python3 prep_wiki.py 4e8 --tokenizer llama --out_prefix data/wiki32k   # Llama 32k

Writes <prefix>_all.bin incrementally (low RAM), then splits off a val tail of
~--val_tokens. The split point is moved forward to an article separator so no
article straddles train/val (article-level split). An end-of-text token
separates articles (GPT-2 <|endoftext|> 50256, Llama </s> 2).
"""
import argparse, time, numpy as np
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("target", type=float, nargs="?", default=1.25e9, help="tokens")
ap.add_argument("--tokenizer", choices=["gpt2", "llama"], default="gpt2")
ap.add_argument("--out_prefix", default="data/wiki")
ap.add_argument("--val_tokens", type=int, default=2_000_000)
ap.add_argument("--batch", type=int, default=512, help="articles per encode batch")
args = ap.parse_args()
TARGET = int(args.target)

if args.tokenizer == "gpt2":
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token
    encode_batch = lambda texts: enc.encode_ordinary_batch(texts)
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    EOT = tok.token_to_id("</s>")
    encode_batch = lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

n = 0; ndoc = 0; t0 = time.time(); nextlog = 50_000_000; buf = []
all_path = args.out_prefix + "_all.bin"


def flush(f, texts):
    global n, ndoc
    for ids in encode_batch(texts):
        ids.append(EOT)
        np.array(ids, dtype=np.uint16).tofile(f)
        n += len(ids); ndoc += 1


with open(all_path, "wb") as f:
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) == args.batch:
            flush(f, buf); buf = []
            if n >= nextlog:
                dt = time.time() - t0
                print(f"{n/1e6:.0f}M tokens | {ndoc:,} docs | {n/dt/1e3:.0f}k tok/s "
                      f"| {dt/60:.1f} min", flush=True)
                nextlog += 50_000_000
            if n >= TARGET:
                break
    if buf and n < TARGET:
        flush(f, buf)

print(f"done streaming: {n:,} tokens, {ndoc:,} docs, {(time.time()-t0)/60:.1f} min",
      flush=True)

# article-level val tail: first separator at/after len - val_tokens
d = np.memmap(all_path, dtype=np.uint16, mode="r")
start = len(d) - args.val_tokens
cut = start + int(np.argmax(d[start:] == EOT)) + 1
d[:cut].tofile(args.out_prefix + "_train.bin")
d[cut:].tofile(args.out_prefix + "_val.bin")
print(f"train {cut:,} | val {len(d)-cut:,} (split at article boundary) -> "
      f"{args.out_prefix}_train.bin, {args.out_prefix}_val.bin", flush=True)
