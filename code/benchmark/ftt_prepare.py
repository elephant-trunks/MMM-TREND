#!/usr/bin/env python3
"""Write our split in the original FTT layout:
ftt_reweight/roll_seasonal_data/<name>/roll_season_k/{train,val,test}.json
Same train/val as every other baseline (train-once <= 2023-Q4, seed 2378); test = 2024-Qk.
Fields as in FTT: content, label (truth/rumor), id, year, season, month, weight (=1 before reweighting).
Usage: python ftt_prepare.py [--smoke]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

SMOKE = "--smoke" in sys.argv
YEAR = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
NAME = "mmmtrend_smoke" if SMOKE else (f"mmmtrend{YEAR}" if YEAR else "mmmtrend")
R = C.load_rows()
tr, va, te = C.year_split(R, YEAR) if YEAR else C.splits(R)
if SMOKE:
    tr, va, te = tr[:400], va[:32], {k: v[:16] for k, v in te.items()}


def rec(i):
    x = R[i]
    return {"content": x["text"], "label": "rumor" if x["y"] == 1 else "truth", "id": i,
            "year": x["yq"][0], "season": x["yq"][1], "month": x["month"], "weight": 1.0}


base = os.path.join(C.ROOT, "ftt_reweight", "roll_seasonal_data", NAME)
for k in range(1, 5):
    d = os.path.join(base, f"roll_season_{k}")
    os.makedirs(d, exist_ok=True)
    for fn, ids in [("train.json", tr), ("val.json", va), ("test.json", te[k])]:
        json.dump([rec(i) for i in ids], open(os.path.join(d, fn), "w"), ensure_ascii=False)
print(f"{NAME}: train={len(tr)} val={len(va)} tests={[len(te[k]) for k in range(1, 5)]} -> {base}")
