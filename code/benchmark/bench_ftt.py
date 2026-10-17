#!/usr/bin/env python3
"""FTT_T Step 2: repo Codes/Baselines/models/bert.py trained with FTT instance weights.
For each 2024 quarter k, weights come from FTT Step 1 output
ftt_reweight/roll_seasonal_user_data/<dataset>/roll_season_k/train.json; one model per
quarter (as in FTT). Model/hyper-params identical to our BERT_T run: frozen mBERT (last layer
trainable) + MaskAttention + MLP[384], InstanceWeightedBCELoss, Adam lr 2e-5 wd 5e-5,
max_len 170, batch 64, epochs<=50, early stop 5 on val macro-F1 @ 0.40, seed 2378.
Usage: python bench_ftt.py [--smoke]"""
import sys, os, json, random, numpy as np, torch, torch.nn as nn
from transformers import BertModel, BertTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

SMOKE = "--smoke" in sys.argv
YEARS = C.parse_years(sys.argv)          # e.g. 2023 or 2023+2024 (quarters pooled)
DATA = "mmmtrend_smoke" if SMOKE else (f"mmmtrend{min(YEARS)}" if YEARS else "mmmtrend")  # pooled years reuse the reweighting of the first test year
DATASET = f"{DATA}_SBERT-sg-0.6_prophet-bsr-3_sw-5.0-0.5-2.0"
THR = 0.40; BERTP = "bert-base-multilingual-cased"; MAXLEN = 170; BATCH = 64; EPOCHS = 50
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


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
tr, va, te = C.year_split(R, YEARS) if YEARS else C.splits(R)
if SMOKE: tr, va, te = tr[:400], va[:32], {k: v[:16] for k, v in te.items()}
tok = BertTokenizer.from_pretrained(BERTP)
def enc(ids):
    t = torch.tensor([tok.encode(R[i]["text"], max_length=MAXLEN, padding="max_length", truncation=True) for i in ids])
    return t, (t != tok.pad_token_id).long()
Xtr, Mtr = enc(tr); ytr = torch.tensor([R[i]["y"] for i in tr], dtype=torch.float)
Xva, Mva = enc(va); yva = [R[i]["y"] for i in va]
wstats, probs_by_q = {}, {}
CKPT = os.path.join(C.ROOT, "bench_results", "ftt_text" + C.year_tag(YEARS) + ("_SMOKE" if SMOKE else "") + "_Q{}_partial.json")
for k in range(1, 5):
    if os.path.exists(CKPT.format(k)):                        # quarter already finished in an earlier run
        d = json.load(open(CKPT.format(k))); probs_by_q[k], wstats[f"Q{k}"] = d["probs"], d["weights"]
        print(f"[Q{k}] loaded from checkpoint, skipping", flush=True); continue
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    wf = os.path.join(C.ROOT, "ftt_reweight", "roll_seasonal_user_data", DATASET, f"roll_season_{k}", "train.json")
    w = {i: 1.0 for i in tr}
    if os.path.exists(wf):
        for rec in json.load(open(wf)): w[int(rec["id"])] = float(rec["weight"])
    else:
        print(f"[Q{k}] WARNING no FTT weight file (no cluster passed the MAPE filter) -> all weights 1", flush=True)
    wt = torch.tensor([w[i] for i in tr], dtype=torch.float)
    wstats[f"Q{k}"] = {"reweighted_rows": int((wt != 1).sum()), "min": float(wt.min()), "max": float(wt.max()), "mean": float(wt.mean())}
    print(f"\n##### FTT Q{k}: weights {wstats[f'Q{k}']}", flush=True)
    model = BERTModel().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-5)
    def probs(X, M):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH): out += model(X[i:i+BATCH].to(DEV), M[i:i+BATCH].to(DEV)).cpu().tolist()
        return out
    best, bad, bs = -1, 0, None
    for ep in range(1 if SMOKE else EPOCHS):
        model.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
        for i in range(0, len(order), BATCH):
            b = order[i:i+BATCH]
            p = model(Xtr[b].to(DEV), Mtr[b].to(DEV))
            loss = (nn.functional.binary_cross_entropy(p, ytr[b].to(DEV), reduction="none") * wt[b].to(DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        vm = C.metr(yva, probs(Xva, Mva), THR)["metric"]; print(f"[Q{k}] ep{ep} val_macroF1@{THR}={vm:.4f}", flush=True)
        if vm > best: best, bad, bs = vm, 0, {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: print(f"[Q{k}] early stop", flush=True); break
    model.load_state_dict(bs)
    C.save_trainable(bs, "ftt_text" + C.year_tag(YEARS) + f"_Q{k}")
    probs_by_q[k] = probs(*enc(te[k]))
    json.dump({"probs": probs_by_q[k], "weights": wstats[f"Q{k}"]}, open(CKPT.format(k), "w"))
    print(f"[Q{k}] saved checkpoint", flush=True)
    del model; torch.mps.empty_cache() if DEV == "mps" else None
out = C.report("ftt_text" + C.year_tag(YEARS) + ("_SMOKE" if SMOKE else ""), THR, R, te, probs_by_q)
json.dump(wstats, open(os.path.join(C.ROOT, "bench_results", "ftt_text" + C.year_tag(YEARS) + ("_SMOKE" if SMOKE else "") + "_weights.json"), "w"), indent=2)
print("ALL DONE", flush=True)
