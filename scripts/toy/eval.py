"""Induction measurements for every snapshot (ckpt_*.pt) of a toy2 run.

Eval sequences: 128 x 256 tokens, 4 random segments of 32, each followed by its repeat. Shuffled control: the
repeat is a random permutation of the segment (same tokens, order destroyed).
  first      loss on the first copies (unpredictable, ~ log V)
  exact      loss on the repeats (targets 2..32 of each repeat)
  shuffled   the same on the shuffled control
  induction  shuffled - exact: the loss the model saves by copying in order
  prev       best head's attention t -> t-1, per layer
  ind        best head's attention from a repeated token to the token after its first occurrence, per layer

  python -m scripts.toy.eval RUN_DIR [kernel|master]
"""
import os, sys, glob, json, re, torch
import torch.nn.functional as F
from scripts.analysis.induction_heads import load, CAP, ON

L, NSEG, B = 32, 4, 128
run, kind = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "kernel")


def seqs(V, shuffle, seed=7):
    g = torch.Generator().manual_seed(seed)
    parts = []
    for _ in range(NSEG):
        s = torch.randint(0, V, (B, L), generator=g)
        r = s[:, torch.randperm(L, generator=g)] if shuffle else s
        parts += [s, r]
    return torch.cat(parts, 1).cuda()                               # [B, 2 * L * NSEG]


T = 2 * L * NSEG
pos = torch.arange(T - 1, device="cuda")
in_rep = ((pos + 1) // L) % 2 == 1                                  # target lies in a repeat
first_of_rep = (pos + 1) % (2 * L) == L                             # first token of a repeat: no cue yet
rep_mask, first_mask = in_rep & ~first_of_rep, ~in_rep
q = torch.arange(T - 1, device="cuda")                              # input positions
q_rep = q[((q // L) % 2 == 1) & (q % L < L - 1)]                    # repeated tokens with a successor in the segment
paths = sorted(glob.glob(f"{run}/ckpt_*.pt"), key=lambda p: int(re.findall(r"ckpt_(\d+)", p)[0]))
out = []
for path in paths:
    s, m, V = load(path, kind)
    V = int(os.environ.get("TOY_V", V))                          # data vocabulary (<= model vocabulary)
    row = {"step": s}
    for name, shuf in (("exact", False), ("shuffled", True)):
        x = seqs(V, shuf)
        CAP.clear(); ON[0] = not shuf
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lp = F.cross_entropy(m(x[:, :-1])[0].float().transpose(1, 2), x[:, 1:], reduction="none")
        ON[0] = False
        row[name] = lp[:, rep_mask].mean().item()
        if not shuf:
            row["first"] = lp[:, first_mask].mean().item()
            A = [a.mean(0) for a in CAP]                            # per layer [H, T, T]
            row["prev"] = [a[:, q[1:], q[1:] - 1].mean(-1).max().item() for a in A]
            row["ind"] = [a[:, q_rep, q_rep - L + 1].mean(-1).max().item() for a in A]
    row["induction"] = row["shuffled"] - row["exact"]
    out.append(row)
    print(f"step {s:5d}  first {row['first']:.3f}  exact {row['exact']:.3f}  shuffled {row['shuffled']:.3f}  "
          f"induction {row['induction']:+.3f}  prev {' '.join(f'{v:.2f}' for v in row['prev'])}  "
          f"ind {' '.join(f'{v:.2f}' for v in row['ind'])}", flush=True)
    del m
json.dump(out, open(f"{run}/toy_eval.json", "w"), indent=1)
