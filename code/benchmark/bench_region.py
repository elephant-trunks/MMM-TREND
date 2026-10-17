#!/usr/bin/env python3
"""Region-wise (leave-one-region-out) benchmark, R1..R4 (see bench_common.REGIONS).
Fold k: train on every row outside region k (Pan-India + International always in train,
10% random val, seed 2378), test on region k. All years pooled.
Models / hyper-params identical to the temporal runs:
  image & VL (cached frozen features): MLP[384] head, Adam lr 2e-5 wd 5e-5, batch 64,
      epochs<=50, early stop 5 on val macro-F1 @ 0.30
  bert / eann: frozen mBERT (last layer trainable), max_len 170, same optimiser, @ 0.40
Each fold is checkpointed (bench_results/<model>_region_R<k>_partial.json); re-running
skips finished folds. Usage: python bench_region.py <model>
  model in: clip_image clip_vl blip_image blip_vl qwen_image qwen_vl bert eann"""
import sys, os, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

MODEL = sys.argv[1]
TEXT = MODEL in ("bert", "eann")
THR = 0.40 if TEXT else 0.30
NAME = f"{MODEL}{'_text' if TEXT else ''}_region"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = os.path.join(C.ROOT, "bench_results", NAME + "_R{}_partial.json")
R = C.load_rows()


def seed_all():
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)


def early_stop_train(step_epoch, val_score, get_state, epochs=50):
    best, bad, bs = -1, 0, None
    for ep in range(epochs):
        step_epoch(ep)
        vm = val_score()
        print(f"  ep{ep} val_macroF1@{THR}={vm:.4f}", flush=True)
        if vm > best: best, bad, bs = vm, 0, get_state()
        else:
            bad += 1
            if bad >= 5: print("  early stop", flush=True); break
    return bs


# ---------------------------------------------------------------- image / VL heads
if not TEXT:
    NEEDS = {"clip_image": ["clip_img"], "clip_vl": ["clip_img", "clip_txt"],
             "blip_image": ["blip_img"], "blip_vl": ["blip_img", "blip_txt"],
             "qwen_image": ["qwen_img"], "qwen_vl": ["qwen_vl"]}[MODEL]
    tr24, va24, te24 = C.splits(R)
    ORDER = tr24 + va24 + te24[1] + te24[2] + te24[3] + te24[4]   # order the cache was written in
    def load(fset):
        d = os.path.join(C.ROOT, "bench_features", fset)
        return np.concatenate([np.load(os.path.join(d, f"chunk_{s:06d}.npy")) for s in range(0, len(ORDER), 512)])
    Fm = np.concatenate([load(f) for f in NEEDS], axis=1).astype(np.float32)
    X = np.zeros_like(Fm); X[ORDER] = Fm

    def run_fold(tr, va, te, k):
        seed_all()
        head = nn.Sequential(nn.Linear(X.shape[1], 384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 1))
        opt = torch.optim.Adam(head.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
        Xtr = torch.tensor(X[tr]); yt = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
        Xva = torch.tensor(X[va]); yva = [R[i]["y"] for i in va]
        def pr(Xs):
            head.eval()
            with torch.no_grad(): return torch.sigmoid(head(Xs).squeeze(1)).numpy().tolist()
        def step(ep):
            head.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
            for i in range(0, len(order), 64):
                b = order[i:i + 64]; loss = bce(torch.sigmoid(head(Xtr[b]).squeeze(1)), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        bs = early_stop_train(step, lambda: C.metr(yva, pr(Xva), THR)["metric"],
                              lambda: {kk: v.clone() for kk, v in head.state_dict().items()})
        head.load_state_dict(bs)
        return pr(torch.tensor(X[te]))

# ---------------------------------------------------------------- BERT / EANN
else:
    from transformers import BertModel, BertTokenizer
    BERTP, MAXLEN, BATCH = "bert-base-multilingual-cased", 170, 64

    class Rev(Function):
        @staticmethod
        def forward(ctx, x, a): ctx.a = a; return x.view_as(x)
        @staticmethod
        def backward(ctx, g): return g.neg() * ctx.a, None

    class MaskAtt(nn.Module):
        def __init__(s): super().__init__(); s.a = nn.Linear(768, 1)
        def forward(s, h, m):
            sc = s.a(h).view(-1, h.size(1)).masked_fill(m == 0, float("-inf"))
            return torch.matmul(torch.softmax(sc, -1).unsqueeze(1), h).squeeze(1)

    def mlp(out=True):
        L = [nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.2)]
        return nn.Sequential(*(L + [nn.Linear(384, 1)] if out else L))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.bert = BertModel.from_pretrained(BERTP, attn_implementation="eager")
            for n, p in s.bert.named_parameters(): p.requires_grad = n.startswith("encoder.layer.11")
            s.att = MaskAtt(); s.mlp = mlp()
            if MODEL == "eann":
                s.dom = nn.Sequential(mlp(False), nn.ReLU(), nn.Linear(384, len(C.YEARMAP)))
        def forward(s, ids, m, a=-1):
            f = s.att(s.bert(ids, attention_mask=m)[0], m)
            out = torch.sigmoid(s.mlp(f).squeeze(1))
            return (out, s.dom(Rev.apply(f, a))) if MODEL == "eann" else (out, None)

    tok = BertTokenizer.from_pretrained(BERTP)
    _enc_cache = {}
    def enc(ids):
        t = torch.tensor([tok.encode(R[i]["text"], max_length=MAXLEN, padding="max_length", truncation=True) for i in ids])
        return t, (t != tok.pad_token_id).long()
    print("tokenising all rows once...", flush=True)
    ALL_IDS, ALL_M = enc(list(range(len(R))))

    def run_fold(tr, va, te, k):
        seed_all()
        model = Net().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
        ytr = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
        dtr = torch.tensor([C.YEARMAP[R[i]["year"]] for i in tr])
        yva = [R[i]["y"] for i in va]
        def probs(ids):
            model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(ids), BATCH):
                    b = ids[i:i + BATCH]; out += model(ALL_IDS[b].to(DEV), ALL_M[b].to(DEV))[0].cpu().tolist()
            return out
        def step(ep):
            model.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
            a = max(2. / (1. + np.exp(-10 * ep / 50)) - 1, 0.1)
            for i in range(0, len(order), BATCH):
                bi = order[i:i + BATCH]; rows = [tr[j] for j in bi]
                p, dp = model(ALL_IDS[rows].to(DEV), ALL_M[rows].to(DEV), a)
                loss = bce(p, ytr[bi].to(DEV))
                if MODEL == "eann": loss = loss + F.nll_loss(F.log_softmax(dp, 1), dtr[bi].to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
        bs = early_stop_train(step, lambda: C.metr(yva, probs(va), THR)["metric"],
                              lambda: {kk: v.cpu().clone() for kk, v in model.state_dict().items()})
        model.load_state_dict(bs)
        C.save_trainable(bs, f"{NAME}_R{k}")
        p = probs(te)
        del model
        if DEV == "mps": torch.mps.empty_cache()
        return p


# ---------------------------------------------------------------- folds
tests, probs_by_r = {}, {}
for k in range(1, 5):
    tr, va, te = C.region_split(R, k)
    tests[k] = te
    if os.path.exists(CKPT.format(k)):
        probs_by_r[k] = json.load(open(CKPT.format(k)))["probs"]
        print(f"[R{k}] loaded from checkpoint, skipping", flush=True); continue
    print(f"\n##### {MODEL} R{k}: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    probs_by_r[k] = run_fold(tr, va, te, k)
    json.dump({"probs": probs_by_r[k]}, open(CKPT.format(k), "w"))
    print(f"[R{k}] saved checkpoint", flush=True)

out = C.report(NAME, THR, R, tests, probs_by_r)   # report labels folds Q1..Q4 -> rename to R1..R4
res = {("R" + key[1:] if key.startswith("Q") else key): v for key, v in out["results"].items()}
out.update(results=res, protocol="leave-one-region-out (R1 South, R2 East, R3 West, R4 North/Central); "
                                   "Pan-India + International always in training; all years")
json.dump(out, open(os.path.join(C.ROOT, "bench_results", f"{NAME}_results.json"), "w"), indent=2)
print("ALL DONE", flush=True)
