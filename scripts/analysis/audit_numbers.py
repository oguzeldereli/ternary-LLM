"""Numbers audit: every RUN_INDEX.md table row that names runs (`run` / `run2`) and gives 4-decimal values in its last
column is checked against the run's logged final validation loss, found locally in
checkpoints/final/RUN/train.log, checkpoints/RUN_lab/night.txt, checkpoints/RUN_myr/*, or checkpoints/RUN/metrics.jsonl.
Prints mismatches (> 6e-5), runs without a local log, and a count of matches.   python scripts/analysis/audit_numbers.py"""
import glob, json, os, re

def final_of(run):
    for p in (f"checkpoints/final/{run}/train.log", f"checkpoints/{run}/train.log"):
        if os.path.exists(p):
            m = re.findall(r"FINAL val loss ([\d.]+)", open(p, errors="ignore").read())
            if m: return float(m[-1]), p
    for p in [f"checkpoints/{run}_lab/night.txt"] + glob.glob(f"checkpoints/{run}_myr/*") + [f"checkpoints/{run}/metrics.jsonl"]:
        if os.path.isfile(p):
            t = open(p, errors="ignore").read()
            m = re.findall(r"FINAL val loss ([\d.]+)", t)
            if m: return float(m[-1]), p
            v = [json.loads(l) for l in t.splitlines() if l.startswith("{") and '"val_loss"' in l]
            fin = [x for x in v if x.get("final")] or [x for x in v if x.get("step") in (9155, 18310)]
            if fin: return float(fin[-1]["val_loss"]), p
    return None, None

ok = bad = miss = 0
for ln, line in enumerate(open("docs/RUN_INDEX.md"), 1):
    if not line.startswith("|") or "`" not in line: continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 3: continue
    runs = re.findall(r"`([A-Za-z0-9_.]+)`", cells[0])
    vals = [float(v) for v in re.findall(r"(?<![\d.+-])([1-9]\.\d{4})(?!\d)", cells[-1])]
    if not runs or not vals or len(runs) != len(vals) and not (len(runs) > 1 and len(vals) >= len(runs)): continue
    for r, v in zip(runs, vals[:len(runs)]):
        f, src = final_of(r)
        if f is None:
            miss += 1; print(f"line {ln}: {r}: no local final log (table {v})")
        elif abs(f - v) > 6e-5:
            bad += 1; print(f"line {ln}: MISMATCH {r}: table {v} vs log {f:.4f} ({src})")
        else:
            ok += 1
print(f"{ok} match, {bad} mismatch, {miss} without a local log")
