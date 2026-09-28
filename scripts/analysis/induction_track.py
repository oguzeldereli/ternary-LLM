"""Ordered-copy (induction) score for every snapshot of a set of text runs, cached so a loop only measures new
snapshots. Score: loss on the first copy minus loss on an exact repeat (copy gain), and the same on a shuffled
repeat; induction = exact - shuffled gain. 256 sequences of 64 random tokens repeated once.

  python -m scripts.analysis.induction_track     -> checkpoints/induction_track.json
"""
import os, glob, re, json, torch, torch.nn.functional as F

RUNS = {  # run dir: (loader kind, adapter kind for the loader or "")
    "curve_master": ("master", ""), "magadd16_qk": ("kernel", "add"), "magadd16_wd_qk": ("kernel", "add"),
    "magadd_full": ("kernel", "add"), "lm_lowrank256_xb2_100M": ("kernel", ""), "nola_lab": ("kernel", ""), "nola_add16": ("kernel", "add"), "nola_then_la": ("kernel", ""), "la_sched": ("kernel", ""), "mech_v1_b131": ("kernel", ""), "mech_user_b131": ("kernel", ""), "mech_user_g0_b131": ("kernel", ""), "mech_v1_s0": ("kernel", ""), "mech_user_s0": ("kernel", ""), "mech_user_g0_s0": ("kernel", ""), "small_step_b131": ("kernel", ""), "accum33_b131": ("kernel", ""), "accum33_s0": ("kernel", ""), "small_step_s0": ("kernel", ""), "small_step8_lab": ("kernel", ""), "small_step8_s0": ("kernel", ""), "select_b131": ("kernel", ""), "rc_b131": ("kernel", ""), "evid3_b131": ("kernel", ""), "rc_s0": ("kernel", ""), "master_q4_b131": ("master", ""), "master_q3_b131": ("master", ""), "master_q2_b131": ("master", ""), "adaptrate_b131": ("kernel", ""), "multibeta_b131": ("kernel", ""), "mech_user_q_b131": ("kernel", ""), "la_sched98": ("kernel", ""), "magadd16_qk_lab": ("kernel", "add"), "magadd16_wd_qk_lab": ("kernel", "add"),
}
CACHE = "checkpoints/induction_track.json"


def score(path, kind):
    from scripts.analysis.induction_heads import load
    s, m, V = load(path, kind)
    out = {}
    for shuf in (False, True):
        g = torch.Generator().manual_seed(99); gains = []
        for _ in range(16):
            r = torch.randint(1000, V - 1000, (16, 64), generator=g).cuda()
            r2 = r[:, torch.randperm(64, generator=g).cuda()] if shuf else r
            x = torch.cat([r, r2], 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lp = F.cross_entropy(m(x[:, :-1])[0].float().transpose(1, 2), x[:, 1:], reduction="none")
            gains.append((lp[:, :63].mean(1) - lp[:, 64:].mean(1)).cpu())
        G = torch.cat(gains)
        out["shuffled" if shuf else "exact"] = G.mean().item()
    del m; torch.cuda.empty_cache()
    return s, out


if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for run, (kind, mag) in RUNS.items():
        os.environ["MAG_KIND"] = mag or "add"
        for p in sorted(glob.glob(f"checkpoints/{run}/ckpt_*.pt"), key=lambda p: int(re.findall(r"ckpt_(\d+)", p)[0])):
            key = f"{run}/{os.path.basename(p)}"
            if key in cache: continue
            try:
                s, o = score(p, kind)
            except Exception as e:           # a snapshot still being written by the sync
                print("skip", key, type(e).__name__); continue
            cache[key] = {"run": run, "step": s, "tokens": (s + 1) * 32768, **o, "induction": o["exact"] - o["shuffled"]}
            print(f"{key}: exact {o['exact']:.3f} shuffled {o['shuffled']:.3f} induction {o['exact'] - o['shuffled']:+.3f}", flush=True)
            json.dump(cache, open(CACHE, "w"), indent=1)
