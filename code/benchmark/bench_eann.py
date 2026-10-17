#!/usr/bin/env python3
"""EANN_T (text-only) for the benchmark tables — repo Codes/Baselines/models/eann.py.
Frozen mBERT (last layer trainable) + MaskAttention; classifier MLP[384] + year-domain
classifier behind gradient reversal. Loss = BCE + NLL(domain). Adam lr 2e-5 wd 5e-5,
max_len 170, batch 64, epochs<=50, early stop 5 on val macro-F1 @ 0.40 (repo utils.py),
alpha = max(2/(1+exp(-10*ep/50))-1, 0.1), seed 2378, bert-base-multilingual-cased.
Usage: python bench_eann.py [--smoke]"""
import sys, os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function
from transformers import BertModel, BertTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

SMOKE = "--smoke" in sys.argv
YEARS = C.parse_years(sys.argv)          # e.g. 2023 or 2023+2024 (quarters pooled)
THR = 0.40; BERTP = "bert-base-multilingual-cased"; MAXLEN = 170; BATCH = 64; EPOCHS = 50
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)


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


class EANN(nn.Module):
    def __init__(s):
        super().__init__()
        s.bert = BertModel.from_pretrained(BERTP, attn_implementation="eager")
        for n, p in s.bert.named_parameters(): p.requires_grad = n.startswith("encoder.layer.11")
        s.att = MaskAtt(); s.cls = mlp(); s.dom = nn.Sequential(mlp(False), nn.ReLU(), nn.Linear(384, len(C.YEARMAP)))
    def forward(s, ids, m, a):
        f = s.att(s.bert(ids, attention_mask=m)[0], m)
        return torch.sigmoid(s.cls(f).squeeze(1)), s.dom(Rev.apply(f, a))


R = C.load_rows()
tr, va, te = C.year_split(R, YEARS) if YEARS else C.splits(R)
if SMOKE: tr, va, te = tr[:64], va[:32], {k: v[:16] for k, v in te.items()}
print(f"EANN: train={len(tr)} val={len(va)} tests={[len(te[k]) for k in range(1, 5)]}", flush=True)
tok = BertTokenizer.from_pretrained(BERTP)
def enc(ids):
    t = torch.tensor([tok.encode(R[i]["text"], max_length=MAXLEN, padding="max_length", truncation=True) for i in ids])
    return t, (t != tok.pad_token_id).long()
Xtr, Mtr = enc(tr); ytr = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
dtr = torch.tensor([C.YEARMAP[R[i]["year"]] for i in tr])
Xva, Mva = enc(va); yva = [R[i]["y"] for i in va]
model = EANN().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
def probs(X, M):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH): out += model(X[i:i+BATCH].to(DEV), M[i:i+BATCH].to(DEV), -1)[0].cpu().tolist()
    return out
best, bad, bs = -1, 0, None
for ep in range(1 if SMOKE else EPOCHS):
    model.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
    a = max(2. / (1. + np.exp(-10 * ep / EPOCHS)) - 1, 0.1)
    for i in range(0, len(order), BATCH):
        b = order[i:i+BATCH]
        p, dp = model(Xtr[b].to(DEV), Mtr[b].to(DEV), a)
        loss = bce(p, ytr[b].to(DEV)) + F.nll_loss(F.log_softmax(dp, 1), dtr[b].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    vm = C.metr(yva, probs(Xva, Mva), THR)["metric"]; print(f"ep{ep} val_macroF1@{THR}={vm:.4f}", flush=True)
    if vm > best: best, bad, bs = vm, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: print("early stop", flush=True); break
model.load_state_dict(bs)
C.save_trainable(bs, "eann_text" + C.year_tag(YEARS))
C.report("eann_text" + C.year_tag(YEARS) + ("_SMOKE" if SMOKE else ""), THR, R, te, {k: probs(*enc(te[k])) for k in range(1, 5)})
print("ALL DONE", flush=True)
