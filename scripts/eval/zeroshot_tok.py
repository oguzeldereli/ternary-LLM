"""Tokenise the zero-shot tasks (data/zeroshot/*.jsonl, from HF: LAMBADA OpenAI, HellaSwag, PIQA, ARC-Easy / Challenge,
WinoGrande) with the training tokenizer (Llama-2 32k), CPU only -> data/zeroshot/tok/TASK.json.
Continuation tokens are taken as enc(context + continuation)[len(enc(context)):] (as lm-evaluation-harness does), and
byte lengths are kept for length-normalised accuracy. A document starts with EOT (2), as the training stream does.
    python scripts/eval/zeroshot_tok.py"""
import json, os
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
enc = lambda s: tok.encode(s, add_special_tokens=False).ids


def split(ctx, cont):
    a = enc(ctx); b = enc(ctx + cont)
    k = len(a)
    while k > 0 and b[:k] != a[:k]:         # a merge across the boundary: back off to the common prefix
        k -= 1
    return [2] + b[:k], b[k:]


os.makedirs("data/zeroshot/tok", exist_ok=True)
for task in ("lambada", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"):
    out = []
    for l in open(f"data/zeroshot/{task}.jsonl"):
        x = json.loads(l)
        if task == "lambada":
            c, t = split(x["context"], x["target"]); out.append({"ctx": c, "cont": [t], "label": 0})
        elif task == "winogrande":            # partial scoring: two contexts, one shared continuation
            pairs = [split(cx, x["target"]) for cx in x["contexts"]]
            out.append({"ctxs": [p[0] for p in pairs], "conts": [p[1] for p in pairs], "label": x["label"]})
        else:
            pairs = [split(x["context"], ch) for ch in x["choices"]]
            out.append({"ctxs": [p[0] for p in pairs], "conts": [p[1] for p in pairs], "label": x["label"],
                        "nbytes": [len(ch.encode("utf-8")) for ch in x["choices"]]})
    json.dump(out, open(f"data/zeroshot/tok/{task}.json", "w"))
    print(task, len(out))
