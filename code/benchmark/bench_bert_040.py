#!/usr/bin/env python3
"""BERT_T on complete_dataset_v1.2.xlsx, faithful 0.40 decision boundary (repo metrics()).
Same model/protocol as before (frozen mBERT last-layer + MaskAttention + MLP[384], BCE,
Adam lr2e-5 wd5e-5, max_len170 batch64 early_stop5 epoch<=50 seed2378, mBERT).
Train once on <=2023-Q4 (val=10%), evaluate each 2024 quarter.
EXTRA: saves per-quarter test probabilities+labels, per-quarter ROC-AUC, and a
threshold sweep (macro-F1 vs threshold) so we can see if 0.40 helps and whether the
Q3/Q4 ceiling is a threshold vs separability problem."""
import openpyxl, re, os, json, collections, random, numpy as np, torch, torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
SEED=2378; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV="mps" if torch.backends.mps.is_available() else "cpu"
BERTP="bert-base-multilingual-cased"; MAXLEN=170; BATCH=64; LR=2e-5; WD=5e-5; EPOCHS=50; PATIENCE=5
THR=0.40
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX=os.path.join(_ROOT,"data","complete_dataset_v1.2.xlsx")
ART=os.path.join(_ROOT,"data","articles_text")
AUG=ART
def cell(v): return "" if v is None else str(v).strip()
def qof(v):
    m=re.match(r"(\d{2})/(\d{2})/(\d{4})",cell(v)); return (int(m[3]),(int(m[2])-1)//3+1) if m else None
def body(File):
    f=cell(File)
    if not f: return ""
    p=os.path.join(AUG if f.startswith("ID-AUG-") else ART, f)
    if os.path.exists(p):
        try: return open(p,encoding="utf-8",errors="ignore").read().strip()
        except: return ""
    return ""
wb=openpyxl.load_workbook(XLSX, read_only=True); ws=wb["Complete-Dataset"]; it=ws.iter_rows(values_only=True)
H=[h for h in next(it) if h is not None]; idx={h:i for i,h in enumerate(H)}
R=[]; nb=0
for r in it:
    q=qof(r[idx["dates"]])
    if not q or not(2012<=q[0]<=2024): continue
    b=body(r[idx["File"]])
    if b and len(b)>=20: txt=b; nb+=1
    else: txt=(cell(r[idx["Title"]])+". "+cell(r[idx["Summary"]])).strip()
    R.append({"text":txt,"auth_raw":cell(r[idx["Authenticity"]]),"yq":q})
wb.close()
print(f"rows={len(R)} body-used={nb} ({100*nb/len(R):.1f}%)  THRESHOLD={THR}",flush=True)
FIXED={"True":"real","Real":"real","False":"fake","Fake":"fake","Problematic":"fake"}
for x in R: x["lab"]=FIXED.get(x["auth_raw"])
qc=collections.defaultdict(lambda:{"real":0,"fake":0}); g={"real":0,"fake":0}
for x in R:
    if x["lab"]: g[x["lab"]]+=1; qc[x["yq"]][x["lab"]]+=1
for x in sorted([x for x in R if x["lab"] is None],key=lambda x:(x["yq"][0],x["yq"][1],x["auth_raw"])):
    c=qc[x["yq"]]; cls="real" if c["real"]<c["fake"] else "fake"; c[cls]+=1; x["lab"]=cls; g[cls]+=1
print(f"real={g['real']} fake={g['fake']}",flush=True)
LMAP={"real":0,"fake":1}
def ordk(yq): return yq[0]*4+yq[1]
class MLP(nn.Module):
    def __init__(s,d,dims,dp):
        super().__init__(); L=[]; inp=d
        for e in dims: L+=[nn.Linear(inp,e),nn.ReLU(),nn.Dropout(dp)]; inp=e
        L.append(nn.Linear(inp,1)); s.mlp=nn.Sequential(*L)
    def forward(s,x): return s.mlp(x)
class MaskAttention(nn.Module):
    def __init__(s,d): super().__init__(); s.a=nn.Linear(d,1)
    def forward(s,inp,mask):
        sc=s.a(inp).view(-1,inp.size(1)).masked_fill(mask==0,float("-inf"))
        sc=torch.softmax(sc,-1).unsqueeze(1); return torch.matmul(sc,inp).squeeze(1)
class BERTModel(nn.Module):
    def __init__(s):
        super().__init__(); s.bert=BertModel.from_pretrained(BERTP,attn_implementation="eager").requires_grad_(False)
        for n,p in s.bert.named_parameters(): p.requires_grad=n.startswith("encoder.layer.11")
        s.mlp=MLP(768,[384],0.2); s.att=MaskAttention(768)
    def forward(s,ids,mask):
        f=s.att(s.bert(ids,attention_mask=mask)[0],mask); return torch.sigmoid(s.mlp(f).squeeze(1))
tok=BertTokenizer.from_pretrained(BERTP)
def encode(texts):
    ids=[tok.encode(t,max_length=MAXLEN,add_special_tokens=True,padding="max_length",truncation=True) for t in texts]
    ids=torch.tensor(ids); return ids,(ids!=tok.pad_token_id).long()
def probs(model,ids,mask):
    model.eval(); pr=[]
    with torch.no_grad():
        for i in range(0,len(ids),BATCH): pr.extend(model(ids[i:i+BATCH].to(DEV),mask[i:i+BATCH].to(DEV)).cpu().numpy().tolist())
    return pr
def metr(y,pr,thr):
    yp=[1 if p>thr else 0 for p in pr]; labs=sorted(set(y)); f1v=f1_score(y,yp,average=None,zero_division=0)
    fr=f1v[labs.index(0)] if 0 in labs else 0.0; fk=f1v[labs.index(1)] if 1 in labs else 0.0
    return {"metric":float(f1_score(y,yp,average="macro",zero_division=0)),"acc":float(accuracy_score(y,yp)),"f1_real":float(fr),"f1_fake":float(fk)}
cut=ordk((2024,1)); tr=[x for x in R if ordk(x["yq"])<cut]
random.Random(SEED).shuffle(tr); nv=int(0.1*len(tr)); val=tr[:nv]; trn=tr[nv:]
tests={k:[x for x in R if x["yq"]==(2024,k)] for k in range(1,5)}
print(f"TRAIN={len(trn)} VAL={len(val)} | "+", ".join(f'Q{k}={len(tests[k])}' for k in range(1,5)),flush=True)
Xtr,Mtr=encode([x["text"] for x in trn]); ytr=torch.tensor([LMAP[x["lab"]] for x in trn],dtype=torch.float)
Xva,Mva=encode([x["text"] for x in val]); yva=[LMAP[x["lab"]] for x in val]
enc_t={k:(encode([x["text"] for x in tests[k]]),[LMAP[x["lab"]] for x in tests[k]]) for k in range(1,5)}
model=BERTModel().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD); bce=nn.BCELoss()
best=-1; bad=0; bs=None; order=list(range(len(trn)))
for ep in range(EPOCHS):
    model.train(); random.Random(SEED+ep).shuffle(order)
    for i in range(0,len(order),BATCH):
        b=order[i:i+BATCH]; p=model(Xtr[b].to(DEV),Mtr[b].to(DEV)); loss=bce(p,ytr[b].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    vm=metr(yva,probs(model,Xva,Mva),THR)["metric"]; print(f"ep{ep} val_macroF1@{THR}={vm:.4f}",flush=True)
    if vm>best: best=vm; bad=0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        bad+=1
        if bad>=PATIENCE: print("early stop",flush=True); break
model.load_state_dict(bs)
results={}; probstore={}; sweep_thr=[round(t,2) for t in np.arange(0.20,0.81,0.05)]
sweep={f"Q{k}":{} for k in range(1,5)}; sweep["Avg"]={}
for k in range(1,5):
    (Xte,Mte),yte=enc_t[k]; pr=probs(model,Xte,Mte)
    probstore[f"Q{k}"]={"y":yte,"p":pr}
    results[f"Q{k}"]=metr(yte,pr,THR)
    try: results[f"Q{k}"]["auc"]=float(roc_auc_score(yte,pr))
    except: results[f"Q{k}"]["auc"]=None
    for t in sweep_thr: sweep[f"Q{k}"][t]=metr(yte,pr,t)["metric"]
    print(f"Q{k}@{THR}: {results[f'Q{k}']}",flush=True)
results["Avg"]={m:float(np.mean([results[f"Q{k}"][m] for k in range(1,5)])) for m in ["metric","acc","f1_fake","f1_real"]}
results["Avg"]["auc"]=float(np.mean([results[f"Q{k}"]["auc"] for k in range(1,5) if results[f"Q{k}"]["auc"] is not None]))
for t in sweep_thr: sweep["Avg"][t]=float(np.mean([sweep[f"Q{k}"][t] for k in range(1,5)]))
json.dump(results,open("bench_bert_040_results.json","w"),indent=2)
json.dump({"threshold":THR,"probs":probstore,"sweep":sweep,"sweep_thresholds":sweep_thr},open("bench_bert_040_probs.json","w"),indent=2)
print(f"\n===== BERT_T @ {THR} (v1.2 2024) =====")
for k in ["Q1","Q2","Q3","Q4","Avg"]:
    r=results[k]; print(f"{k}: MacroF1={r['metric']:.4f} Acc={r['acc']:.4f} F1fake={r['f1_fake']:.4f} F1real={r['f1_real']:.4f} AUC={r.get('auc')}",flush=True)
print("\n=== threshold sweep (macro-F1) ===")
print("thr   "+"  ".join(f"Q{k}" for k in range(1,5))+"   Avg")
for t in sweep_thr:
    print(f"{t:.2f}  "+"  ".join(f"{sweep[f'Q{k}'][t]:.3f}" for k in range(1,5))+f"   {sweep['Avg'][t]:.3f}",flush=True)
print("\nALL DONE",flush=True)
