#!/usr/bin/env python3
"""5-seed robustness: encode a pool once (CLIP+BLIP), then resample 1500 and
recluster over 5 seeds. Report NPMI + silhouette as mean +/- std per arm."""
import openpyxl, os, re, random, math, collections, numpy as np, torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMGDIR=os.path.join(_ROOT,"images")
DEV="mps" if torch.backends.mps.is_available() else "cpu"; K=7; POOL=2500; SUB=1500; SEEDS=5
def isempty(v): return v is None or str(v).strip() in ("","None","none","nan")
random.seed(0); np.random.seed(0); torch.manual_seed(0)

wb=openpyxl.load_workbook(os.path.join(_ROOT,"data","complete_dataset_v1.2.xlsx"), read_only=True)
ws=wb["Complete-Dataset"]; it=ws.iter_rows(values_only=True)
H=[h for h in next(it) if h is not None]; idx={h:i for i,h in enumerate(H)}
cand=[]
for r in it:
    img=r[idx["Best Match Image (BLIP)"]]; txt=r[idx["Summary(EN)"]]
    if isempty(img) or isempty(txt): continue
    p=os.path.join(IMGDIR,str(img).strip())
    if os.path.exists(p): cand.append((p,str(txt)))
wb.close()
random.shuffle(cand); cand=cand[:POOL]
paths=[c[0] for c in cand]; texts=[c[1] for c in cand]
pil=[]
for p in paths:
    try: pil.append(Image.open(p).convert("RGB"))
    except: pil.append(Image.new("RGB",(224,224)))
print(f"pool: {len(cand)}",flush=True)
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-9)

print("CLIP...",flush=True)
clip=SentenceTransformer("clip-ViT-B-32")
t=l2(clip.encode(texts,batch_size=64,convert_to_numpy=True)).astype(np.float32)
im=l2(clip.encode(pil,batch_size=32,convert_to_numpy=True)).astype(np.float32)

print("BLIP cross-attention...",flush=True)
from transformers import BlipProcessor, BlipForImageTextRetrieval
proc=BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip=BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(DEV).eval()
bz=[]
with torch.no_grad():
    for s in range(0,len(pil),16):
        enc=proc(images=pil[s:s+16],text=[x[:300] for x in texts[s:s+16]],return_tensors="pt",padding=True,truncation=True,max_length=40).to(DEV)
        v=blip.vision_model(enc.pixel_values)[0]; a=torch.ones(v.size()[:-1],dtype=torch.long,device=DEV)
        o=blip.text_encoder(input_ids=enc.input_ids,attention_mask=enc.attention_mask,encoder_hidden_states=v,encoder_attention_mask=a,return_dict=True)
        bz.append(o.last_hidden_state[:,0,:].cpu().numpy())
        if s%320==0: print(f"  BLIP {s}/{len(pil)}",flush=True)
bz=l2(np.concatenate(bz,0).astype(np.float32))

def gated(t,i,epochs=250):
    T=torch.tensor(t); I=torch.tensor(i); d=t.shape[1]
    g=torch.nn.Sequential(torch.nn.Linear(2*d,d),torch.nn.Sigmoid()); dec=torch.nn.Linear(d,2*d)
    opt=torch.optim.Adam(list(g.parameters())+list(dec.parameters()),lr=1e-3); X=torch.cat([T,I],1)
    for e in range(epochs):
        a=g(X); z=a*T+(1-a)*I; loss=((dec(z)-X)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): a=g(X); return l2((a*T+(1-a)*I).numpy())

STOP=set("the a an and or of to in on for is are was were be been being this that these those with as at by from it its their his her they he she we you i not no can will would video image viral claim claims social media fact check shared people".split())
def tok(s): return [w for w in re.findall(r"[a-zA-Z]{3,}",str(s).lower()) if w not in STOP]
def coherence(lab,txts):
    docs=[set(tok(x)) for x in txts]; Nd=len(docs); df=collections.Counter()
    for d in docs:
        for w in d: df[w]+=1
    def npmi(w1,w2):
        d1=df[w1]/Nd; d2=df[w2]/Nd; co=sum(1 for d in docs if w1 in d and w2 in d)/Nd
        return -1.0 if co==0 else (math.log(co/(d1*d2)))/(-math.log(co))
    sc=[]
    for k in set(lab):
        cd=[txts[j] for j in range(len(txts)) if lab[j]==k]; wc=collections.Counter()
        for x in cd:
            for w in set(tok(x)): wc[w]+=1
        top=[w for w,_ in wc.most_common(10)]
        pr=[(top[a],top[b]) for a in range(len(top)) for b in range(a+1,len(top))]
        if pr: sc.append(np.mean([npmi(*p) for p in pr]))
    return float(np.mean(sc)) if sc else float("nan")

res={a:{"npmi":[],"sil":[]} for a in ["mean","concat","gated","BLIP-xattn"]}
for seed in range(SEEDS):
    rng=np.random.RandomState(seed); sidx=rng.choice(len(cand),SUB,replace=False)
    tt=t[sidx]; ii=im[sidx]; bb=bz[sidx]; txts=[texts[j] for j in sidx]
    Z={"mean":l2((tt+ii)/2),"concat":l2(np.concatenate([tt,ii,np.abs(tt-ii),tt*ii],1)),
       "gated":gated(tt,ii),"BLIP-xattn":bb}
    for a,z in Z.items():
        lab=KMeans(n_clusters=K,n_init=10,random_state=seed).fit(z).labels_
        res[a]["npmi"].append(coherence(lab,txts)); res[a]["sil"].append(silhouette_score(z,lab))
    print(f"seed {seed} done",flush=True)

print("\n%-12s %-18s %-18s"%("ARM","NPMI (mean±std)","silhouette (mean±std)"),flush=True)
for a in ["mean","concat","gated","BLIP-xattn"]:
    n=res[a]["npmi"]; s=res[a]["sil"]
    print("%-12s %.4f ± %.4f     %.4f ± %.4f"%(a,np.mean(n),np.std(n),np.mean(s),np.std(s)),flush=True)
