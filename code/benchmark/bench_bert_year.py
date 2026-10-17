#!/usr/bin/env python3
"""BERT_T for a chosen test year (same model/hyper-params as the 2024 BERT_T run):
frozen mBERT (last layer trainable) + MaskAttention + MLP[384], BCE, Adam lr 2e-5 wd 5e-5,
max_len 170, batch 64, epochs<=50, early stop 5 on val macro-F1 @ 0.40, seed 2378.
Train on all rows dated before YEAR (10% random val), test each quarter of YEAR.
Usage: python bench_bert_year.py <YEAR>"""
import sys, os, json, random, numpy as np, torch, torch.nn as nn
from transformers import BertModel, BertTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

YEARS = C.parse_years(sys.argv); YEAR = min(YEARS)
THR = 0.40; BERTP = "bert-base-multilingual-cased"; MAXLEN = 170; BATCH = 64; EPOCHS = 50
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)


class MaskAtt(nn.Module):
    def __init__(s): super().__init__(); s.a = nn.Linear(768, 1)
    def forward(s, h, m):
        sc = s.a(h).view(-1, h.size(1)).masked_fill(m == 0, float("-inf"))
        return torch.matmul(torch.softmax(sc, -1).unsqueeze(1), h).squeeze(1)


class BERTModel(nn.Module):
    def __init__(s):
        super().__init__()
        s.bert = BertModel.from_pretrained(BERTP, attn_implementation="eager")
        for n, p in s.bert.named_parameters(): p.requires_grad = n.startswith("encoder.layer.11")
        s.att = MaskAtt(); s.mlp = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 1))
    def forward(s, ids, m):
        return torch.sigmoid(s.mlp(s.att(s.bert(ids, attention_mask=m)[0], m)).squeeze(1))


R = C.load_rows()
tr, va, te = C.year_split(R, YEARS)
print(f"BERT test={YEARS}: train={len(tr)} val={len(va)} tests={[len(te[k]) for k in range(1, 5)]}", flush=True)
tok = BertTokenizer.from_pretrained(BERTP)
def enc(ids):
    t = torch.tensor([tok.encode(R[i]["text"], max_length=MAXLEN, padding="max_length", truncation=True) for i in ids])
    return t, (t != tok.pad_token_id).long()
Xtr, Mtr = enc(tr); ytr = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
Xva, Mva = enc(va); yva = [R[i]["y"] for i in va]
model = BERTModel().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
def probs(X, M):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH): out += model(X[i:i+BATCH].to(DEV), M[i:i+BATCH].to(DEV)).cpu().tolist()
    return out
best, bad, bs = -1, 0, None
for ep in range(EPOCHS):
    model.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
    for i in range(0, len(order), BATCH):
        b = order[i:i+BATCH]; loss = bce(model(Xtr[b].to(DEV), Mtr[b].to(DEV)), ytr[b].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    vm = C.metr(yva, probs(Xva, Mva), THR)["metric"]; print(f"ep{ep} val_macroF1@{THR}={vm:.4f}", flush=True)
    if vm > best: best, bad, bs = vm, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: print("early stop", flush=True); break
model.load_state_dict(bs)
name = "bert_text" + C.year_tag(YEARS)
C.save_trainable(bs, name)
out = C.report(name, THR, R, te, {k: probs(*enc(te[k])) for k in range(1, 5)})
out["protocol"] = f"train on all rows < {YEAR} (10% val), test each quarter pooled over {YEARS}"
json.dump(out, open(os.path.join(C.ROOT, "bench_results", f"{name}_results.json"), "w"), indent=2)
print("ALL DONE", flush=True)
