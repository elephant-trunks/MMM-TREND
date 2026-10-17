#!/usr/bin/env python3
"""Topic induction: mean-fusion CLIP embeddings of every instance, clustered with KMeans (K=7).
Text = English summary (fallback: title); image = best-matched image when available in images/.
The joint vector is the mean of the L2-normalised text and image embeddings (text only when
no image is found). Writes the embeddings and a table of cluster sizes against the released
`Topics` column to topics_out/. The dataset file is not modified."""
import openpyxl, os, collections, numpy as np, torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from PIL import Image
np.random.seed(0); torch.manual_seed(0)
_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX=os.path.join(_ROOT,"data","complete_dataset_v1.2.xlsx"); IMGDIR=os.path.join(_ROOT,"images")
OUT=os.path.join(_ROOT,"topics_out"); K=7
def isempty(v): return v is None or str(v).strip() in ("","None","none","nan")

# image index
bmap={}
if os.path.isdir(IMGDIR):
    for f in os.listdir(IMGDIR):
        if f.lower().endswith((".jpg",".jpeg",".png")): bmap[f]=os.path.join(IMGDIR,f)
print("image basenames indexed:",len(bmap),flush=True)

wb=openpyxl.load_workbook(XLSX, read_only=True)
ws=wb["Complete-Dataset"]; it=ws.iter_rows(values_only=True)
H=[h for h in next(it) if h is not None]; idx={h:i for i,h in enumerate(H)}
imgcols=["Best Match Image (BLIP)","Best Match Image (CLIP)"]+[f"Image {i} ID" for i in range(1,11)]
texts=[]; imgpaths=[]; names=[]
for r in it:
    texts.append(str(r[idx["Summary(EN)"]] or r[idx["Title"]] or ""))
    names.append(str(r[idx["Topics"]] or "").strip())
    ip=None
    for cn in imgcols:
        v=r[idx[cn]]
        if v and str(v).strip() in bmap: ip=bmap[str(v).strip()]; break
    imgpaths.append(ip)
wb.close()
N=len(texts); nimg=sum(1 for p in imgpaths if p)
print(f"rows={N}  with image={nimg}",flush=True)

def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-9)
clip=SentenceTransformer("clip-ViT-B-32")
print("encoding text...",flush=True)
t=l2(clip.encode(texts,batch_size=64,show_progress_bar=False,convert_to_numpy=True)).astype(np.float32)
print("encoding images...",flush=True)
img_emb=np.zeros_like(t)
have=np.zeros(N,bool)
batch=[]; bidx=[]
def flush(batch,bidx):
    if not batch: return
    e=l2(clip.encode(batch,batch_size=32,show_progress_bar=False,convert_to_numpy=True)).astype(np.float32)
    for j,k in enumerate(bidx): img_emb[k]=e[j]; have[k]=True
for i,p in enumerate(imgpaths):
    if p:
        try: batch.append(Image.open(p).convert("RGB")); bidx.append(i)
        except: pass
    if len(batch)>=256: flush(batch,bidx); batch=[]; bidx=[]; print(f"  imgs {i}/{N}",flush=True)
flush(batch,bidx)

# mean fusion where image present, else text-only
z=t.copy()
z[have]=(t[have]+img_emb[have])/2
z=l2(z)
os.makedirs(OUT,exist_ok=True); np.save(os.path.join(OUT,"topics_z.npy"),z)
print("clustering...",flush=True)
km=KMeans(n_clusters=K,n_init=10,random_state=0).fit(z)
lab=km.labels_


np.save(os.path.join(OUT,"cluster_labels.npy"),lab)
tab=collections.Counter(zip(lab.tolist(),names))
with open(os.path.join(OUT,"cluster_vs_topic.csv"),"w",encoding="utf-8") as f:
    f.write("cluster,topic,instances\n")
    for (c,n),v in sorted(tab.items(),key=lambda x:(x[0][0],-x[1])): f.write(f'{c},"{n}",{v}\n')
print("cluster sizes:",sorted(collections.Counter(lab.tolist()).items()),flush=True)
