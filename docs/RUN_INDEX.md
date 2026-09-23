# Run index

Every directory under `checkpoints/` (gitignored). Unless noted: 110M (`small`), wiki32k,
seq 2048, 16 x 2048 = 32,768 tokens/step, same seed and batch order, kernel mode with
int8 forward. "val" = final validation perplexity. Details and discussion: [RUNS.md](RUNS.md).
Each dir has `metrics.jsonl` (+ `ckpt.pt`); logs are `train.log` in the dir, or
`checkpoints/<name>.log` for older runs.

## Ceiling

| run | what | tokens | val |
|---|---|---|---|
| `p2_baseline` | **master weights** (fp32 latent + STE + AdamW, LR 1.5e-3) | 300M | **15.8** |
| `probe_master` | master, 161 steps with probes (`--probe`) | 5M | 326.9 |

## Stateless flips, full length (all bf16 float tail = frozen norm gains)

| run | what | tokens | val |
|---|---|---|---|
| `armA_cosine` | **reference**: flip rate 0.02 cosine-annealed to 0 | 300M | **98.6** |
| `armA_cos_600M` | reference, 600M-token schedule | 600M | 92.7 |
| `armA_cos_r005` | cosine from 4x lower rate (0.005) | 300M | 101.0 |
| `armA_linear` | linear anneal | 300M | 101.4 |
| `armA_cos_lockout` | cosine + per-weight flip lockout | 300M | 102.7 |
| `armA_cos_floor` | cosine to a 0.02% floor | 300M | 104.0 |
| `p3b_acc1` | constant rate | 300M | 135.8 |
| `p3b_acc4` | constant rate, 4x batch | 400M | 135.5 |
| `armA_cos_ef` | cosine + spatial error feedback | 300M | 156.0 |
| `armB_absscale` | frozen absolute threshold | 300M | 160.7 |
| **`armA_cos_la1`** | **look-ahead filter** (`--lookahead 1`), reference schedule; stopped at step 6171, resumable | 202M | **65.3** |

## Short probes and screens (1000 steps or fewer)

| run | what | tokens | val |
|---|---|---|---|
| `lr_ctl` | control for the 1000-step probes | 33M | 156.8 |
| `armA_cos_lr15` | 1.5x float-tail LR | 34M | 153.3 |
| `lo_once_t200`, `lo_once_t1000`, `lo_norev_t1000` | lockout variants | 33M | 175-191 |
| `ef_probe_a0*` | error-feedback strength probes | 20M | 256-828 |
| `gref1/7/10/25/100`, `gref10_r3x` | g_ref sweep (300-step alpha screens) | 10M | 350-490 |
| `rownorm`, `invmag`, `hump` | flip-rule shape screens | 10M | 350-431 |
| `ev3_screen` | 3-bit evidence counter (not stateless) | 10M | 554 |
| `rs_smoke` | greedy flip-rate search | 5M | 633 |
| `la1`, `la2` | look-ahead 1 and 2 passes, alpha screens | 10M | 280, 254 |
| `la_rwarm` | look-ahead, rate warmup 0.02 -> 0.04 (stopped at 790; snapshots every 100) | 26M | - |
| `la_r0ramp` | look-ahead, rate ramp 0 -> 0.02 over 305 steps (stopped at 313) | 10M | - |
| `la_fastramp_lr15` | look-ahead, rate ramp over 30 steps + master's tail LR (snapshots 100/200/300) | 10M | 180.7 |
| `la_fast_fp32tail` | same + **fp32 float tail** (`--tail_fp32`), probes | 10M | 179.2 |
| `la_fast_fp32tail_r04` | same, peak rate 0.04 | 10M | (running) |
| `probe_fast` | `la_fastramp_lr15` config, 161 steps with probes | 5M | 317.3 |

## Scratch / smoke tests (safe to delete)

`acc_a`, `acc_b`, `p2_master`, `sanity_master`, `smoke`, `t_armA`, `t_armB`,
`la_r0ramp_lr15` (stopped at step 20), `p3b_acc16`, `p3b_acc64` (never ran),
`wiki_ev`, `wiki_ev3`, `wiki_mf` (early 51M/GPT-2-vocab runs from RESULTS.md, checkpoint only).
