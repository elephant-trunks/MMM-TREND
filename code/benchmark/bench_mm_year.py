#!/usr/bin/env python3
"""Re-run a cached image/VL baseline with a different test year (no feature extraction).
Protocol: train on all rows dated before YEAR (10% random val, seed 2378), test each quarter of YEAR.
Head/hyper-params identical to bench_mm.py (MLP[384], BCE, Adam lr 2e-5 wd 5e-5, batch 64,
epochs<=50, early stop 5 on val macro-F1 @ 0.30).
Usage: python bench_mm_year.py <clip_image|clip_vl|blip_image|blip_vl|qwen_image|qwen_vl> <YEAR>"""
import sys, os, json, random, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

MODEL = sys.argv[1]
YEARS = [int(y) for y in sys.argv[2].split("+")]   # e.g. "2023" or "2023+2024" (quarters pooled across years)
YEAR = min(YEARS)
THR = 0.30
NEEDS = {"clip_image": ["clip_img"], "clip_vl": ["clip_img", "clip_txt"],
         "blip_image": ["blip_img"], "blip_vl": ["blip_img", "blip_txt"],
         "qwen_image": ["qwen_img"], "qwen_vl": ["qwen_vl"]}[MODEL]
torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)

R = C.load_rows()
# the cache was written in the 2024-split order: train + val + Q1..Q4 (covers every row)
tr24, va24, te24 = C.splits(R)
ORDER = tr24 + va24 + te24[1] + te24[2] + te24[3] + te24[4]
assert sorted(ORDER) == list(range(len(R)))
def load(fset):
    d = os.path.join(C.ROOT, "bench_features", fset)
    return np.concatenate([np.load(os.path.join(d, f"chunk_{s:06d}.npy")) for s in range(0, len(ORDER), 512)])
F = np.concatenate([load(f) for f in NEEDS], axis=1).astype(np.float32)
X = np.zeros_like(F); X[ORDER] = F                  # X[row index in R]

tr = [i for i, x in enumerate(R) if x["yq"][0] < YEAR]
random.Random(C.SEED).shuffle(tr)
nv = int(0.1 * len(tr)); va, tr = tr[:nv], tr[nv:]
te = {k: [i for i, x in enumerate(R) if x["yq"][0] in YEARS and x["yq"][1] == k] for k in range(1, 5)}
print(f"{MODEL} test={YEARS}: train={len(tr)} val={len(va)} tests={[len(te[k]) for k in range(1, 5)]}", flush=True)


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(d, 384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 1))
    def forward(s, x): return torch.sigmoid(s.m(x).squeeze(1))


Xtr = torch.tensor(X[tr]); yt = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
Xva = torch.tensor(X[va]); yva = [R[i]["y"] for i in va]
head = MLP(X.shape[1]); opt = torch.optim.Adam(head.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
def pr(Xs):
    head.eval()
    with torch.no_grad(): return head(Xs).numpy().tolist()
best, bad, bs = -1, 0, None
for ep in range(50):
    head.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
    for i in range(0, len(order), 64):
        b = order[i:i + 64]; loss = bce(head(Xtr[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    vm = C.metr(yva, pr(Xva), THR)["metric"]
    if vm > best: best, bad, bs = vm, 0, {k: v.clone() for k, v in head.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: print(f"early stop ep{ep}", flush=True); break
head.load_state_dict(bs)
name = f"{MODEL}_" + "_".join(map(str, YEARS))
out = C.report(name, THR, R, te, {k: pr(torch.tensor(X[te[k]])) for k in range(1, 5)})
out["protocol"] = f"train on all rows < {YEAR} (10% val), test each quarter pooled over {YEARS}"
json.dump(out, open(os.path.join(C.ROOT, "bench_results", f"{name}_results.json"), "w"), indent=2)
print("ALL DONE", flush=True)
