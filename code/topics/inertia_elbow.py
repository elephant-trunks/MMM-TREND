#!/usr/bin/env python3
"""Elbow evidence: KMeans inertia (and silhouette) over K=2..20 on the mean-fused
text+image embeddings, with the geometric-elbow K marked. Saves a plot + prints values."""
import openpyxl, os, random, numpy as np, torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
random.seed(0); np.random.seed(0)
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMGDIR=os.path.join(_ROOT,"images")
NPOOL=4000
def isempty(v): return v is None or str(v).strip() in ("","None","none","nan")

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
random.shuffle(cand); cand=cand[:NPOOL]
paths=[c[0] for c in cand]; texts=[c[1] for c in cand]
pil=[]
for p in paths:
    try: pil.append(Image.open(p).convert("RGB"))
    except: pil.append(Image.new("RGB",(224,224)))
print(f"rows: {len(cand)}",flush=True)
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-9)
clip=SentenceTransformer("clip-ViT-B-32")
t=l2(clip.encode(texts,batch_size=64,convert_to_numpy=True))
i=l2(clip.encode(pil,batch_size=32,convert_to_numpy=True))
z=l2(((t+i)/2)).astype(np.float32)   # mean fusion (paper's method)

Ks=list(range(2,21)); inertia=[]; sil=[]
for k in Ks:
    km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(z)
    inertia.append(km.inertia_); sil.append(silhouette_score(z,km.labels_))
    print(f"K={k} inertia={km.inertia_:.1f} sil={sil[-1]:.4f}",flush=True)

# geometric elbow (max distance from chord joining endpoints)
x=np.array(Ks,float); y=np.array(inertia,float)
x1,y1=x[0],y[0]; x2,y2=x[-1],y[-1]
d=np.abs((y2-y1)*x-(x2-x1)*y+x2*y1-y2*x1)/np.hypot(y2-y1,x2-x1)
elbowK=int(x[np.argmax(d)])
print(f"\nGEOMETRIC ELBOW at K={elbowK}",flush=True)

fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].plot(Ks,inertia,'o-',color="#2b6cb0"); ax[0].axvline(elbowK,ls='--',color='#e53e3e')
ax[0].set_xlabel("Number of clusters K"); ax[0].set_ylabel("Inertia (WCSS)")
ax[0].set_title(f"Elbow (geometric elbow: K={elbowK})"); ax[0].set_xticks(Ks[::2])
ax[1].plot(Ks,sil,'s-',color="#2f855a"); ax[1].axvline(7,ls='--',color='#e53e3e')
ax[1].set_xlabel("Number of clusters K"); ax[1].set_ylabel("Silhouette")
ax[1].set_title("Silhouette vs K"); ax[1].set_xticks(Ks[::2])
plt.tight_layout(); plt.savefig("elbow_curve.png",dpi=170)
print("saved elbow_curve.png",flush=True)
print("\nK,inertia,silhouette")
for k,ii,ss in zip(Ks,inertia,sil): print(f"{k},{ii:.1f},{ss:.4f}")
