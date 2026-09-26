"""Any-window move analysis from a --record_sample recording (per-step gradient, trit and momentum on a
fixed sample of the ternary weights).

For each window length W (steps) and each non-overlapping window [s, s+W): D = trit(s+W-1) - trit(s-1)
(net move over the window), S = sum of the gradients of steps s..s+W-1, m0 = momentum after step s-1.
Per window: moved share; share of moved trits with D * S < 0 (along the summed gradient); cos(D, -S);
share with D * m0 < 0 (along the momentum); and the same agreement for every single step inside the
window (per-step moves vs that step's own gradient) for reference.

  python -m scripts.analysis.windows_offline RUN_DIR [W1,W2,...] > out.json
"""
import sys, glob, json, numpy as np


def load(run):
    fs = sorted(glob.glob(f"{run}/record/chunk_*.npz"))
    st, g, t, m = [], [], [], []
    for f in fs:
        c = np.load(f)
        st.append(c["step"]); t.append(c["t"])
        g.append((c["g"].view(np.uint16).astype(np.uint32) << 16).view(np.float32))
        m.append((c["m"].view(np.uint16).astype(np.uint32) << 16).view(np.float32))
    return np.concatenate(st), np.concatenate(g), np.concatenate(t).astype(np.int8), np.concatenate(m)


def windows(steps, g, t, m, W):
    out = []
    for a in range(1, len(steps) - W + 1, W):       # need the trit/momentum after step a-1
        D = t[a + W - 1].astype(np.int16) - t[a - 1]
        S = g[a:a + W].sum(0)
        mv = D != 0
        n = int(mv.sum())
        if n == 0:
            continue
        cosDS = float(-(D * S).sum() / (np.linalg.norm(D) * np.linalg.norm(S) + 1e-30))
        out.append({"step": int(steps[a + W - 1]), "moved": n / D.size,
                    "agree_S": float(((D * S)[mv] < 0).mean()), "cos_S": cosDS,
                    "agree_M": float(((D * m[a - 1])[mv] < 0).mean())})
    return out


if __name__ == "__main__":
    run = sys.argv[1]
    Ws = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,5,10,20,50,100,200,500").split(",")]
    steps, g, t, m = load(run)
    print(f"{run}: {len(steps)} steps ({steps[0]}..{steps[-1]}), {g.shape[1]} sampled weights", file=sys.stderr)
    res = {W: windows(steps, g, t, m, W) for W in Ws}
    for W in Ws:
        r = res[W]
        if r:
            print(f"W={W:4d}: {len(r):5d} windows | along sum(g) {np.mean([x['agree_S'] for x in r]):.3f} | "
                  f"cos {np.mean([x['cos_S'] for x in r]):.3f} | along momentum {np.mean([x['agree_M'] for x in r]):.3f} | "
                  f"moved {np.mean([x['moved'] for x in r]) * 100:.2f}%", file=sys.stderr)
    json.dump({str(W): r for W, r in res.items()}, sys.stdout)
