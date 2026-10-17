#!/usr/bin/env python3
"""Image-only / image+text baselines for the benchmark tables (repo Codes/Baselines/models/*).
All backbones in the repo are fully frozen and only MLP[384] is trained, so features are
extracted ONCE, cached, and the head is trained on the cache (same result as running the
repo code, much faster).
  clip_image : CLIP ViT-B/32 image features (512)                -> MLP
  clip_vl    : concat(CLIP image 512, CLIP text 512, max_len 77) -> MLP
  blip_image : BLIP vision last_hidden mean (768)                -> MLP
  blip_vl    : concat(BLIP vision mean 768, BLIP text mean 768)  -> MLP
  qwen_image : Qwen2-VL-2B last hidden mean, prompt '<|image_pad|>'           -> MLP
  qwen_vl    : Qwen2-VL-2B last hidden mean, '<|image_pad|> '+text[:256], 224px -> MLP
Images: Resize(224,224)+ToTensor then back to uint8 PIL (repo dataloader); missing -> black.
Head: MLP[384] dropout .2, sigmoid, BCE, Adam lr 2e-5 wd 5e-5, batch 64, epochs<=50,
early stop 5 on val macro-F1 @ 0.30 (repo utils-Image.py threshold), seed 2378.
Usage: python bench_mm.py <model> [--smoke]"""
import sys, os, json, random, numpy as np, torch, torch.nn as nn
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as C

MODEL = sys.argv[1]
SMOKE = "--smoke" in sys.argv
THR = 0.30
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
FEAT_DIR = os.path.join(C.ROOT, "bench_features_smoke" if SMOKE else "bench_features")
NEEDS = {"clip_image": ["clip_img"], "clip_vl": ["clip_img", "clip_txt"],
         "blip_image": ["blip_img"], "blip_vl": ["blip_img", "blip_txt"],
         "qwen_image": ["qwen_img"], "qwen_vl": ["qwen_vl"]}[MODEL]
tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])


def pil_224(path):
    t = torch.zeros(3, 224, 224)
    if path:
        try:
            t = tf(Image.open(path).convert("RGB"))
        except Exception:
            pass
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8"))


def as_tensor(o):
    return o if torch.is_tensor(o) else getattr(o, "pooler_output", None) if getattr(o, "pooler_output", None) is not None else o[0]


# ---------------- feature extractors (batch of rows -> np.array) ----------------
_cache = {}
def get(kind):
    if kind in _cache:
        return _cache[kind]
    if kind == "clip":
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif kind == "blip":
        from transformers import BlipModel, BlipProcessor
        torch.manual_seed(C.SEED)
        m = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(DEV).eval()
        p = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        m = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True).to(DEV).eval()
        p = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
    _cache[kind] = (m, p)
    return m, p


@torch.no_grad()
def feats(fset, rows):
    if fset == "clip_img":
        m, p = get("clip")
        x = p(images=[pil_224(r["img"]) for r in rows], return_tensors="pt")
        return as_tensor(m.get_image_features(pixel_values=x["pixel_values"].to(DEV))).float().cpu().numpy()
    if fset == "clip_txt":
        m, p = get("clip")
        x = p(text=[r["text"] for r in rows], padding=True, truncation=True, max_length=77, return_tensors="pt")
        return as_tensor(m.get_text_features(input_ids=x["input_ids"].to(DEV),
                                             attention_mask=x["attention_mask"].to(DEV))).float().cpu().numpy()
    if fset == "blip_img":
        m, p = get("blip")
        x = p(images=[pil_224(r["img"]) for r in rows], return_tensors="pt")
        return m.vision_model(pixel_values=x["pixel_values"].to(DEV)).last_hidden_state.mean(1).float().cpu().numpy()
    if fset == "blip_txt":
        m, p = get("blip")
        x = p.tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=512, return_tensors="pt")
        return m.text_model(input_ids=x["input_ids"].to(DEV),
                            attention_mask=x["attention_mask"].to(DEV)).last_hidden_state.mean(1).float().cpu().numpy()
    # Qwen2-VL: run per-sample (no padding) so the mean is over that sample's real tokens only
    m, p = get("qwen")
    out = []
    for r in rows:
        txt = "<|image_pad|>" if fset == "qwen_img" else "<|image_pad|> " + r["text"][:256]
        x = p(text=[txt], images=[pil_224(r["img"])], return_tensors="pt")
        x = {k: (v.to(DEV, dtype=torch.float16) if k == "pixel_values" else v.to(DEV)) for k, v in x.items()}
        h = m(**x, output_hidden_states=True).hidden_states[-1]
        out.append(h.mean(1).float().cpu().numpy()[0])
    return np.stack(out)


def build_cache(fset, R, ids, bs):
    d = os.path.join(FEAT_DIR, fset)
    os.makedirs(d, exist_ok=True)
    CH = 512
    for s in range(0, len(ids), CH):
        f = os.path.join(d, f"chunk_{s:06d}.npy")
        if os.path.exists(f):
            continue
        part = ids[s:s + CH]
        arr = np.concatenate([feats(fset, [R[i] for i in part[j:j + bs]]) for j in range(0, len(part), bs)])
        np.save(f, arr)
        print(f"  [{fset}] {min(s + CH, len(ids))}/{len(ids)}", flush=True)
    return np.concatenate([np.load(os.path.join(d, f"chunk_{s:06d}.npy")) for s in range(0, len(ids), CH)])


# ---------------- run ----------------
R = C.load_rows()
tr, va, te = C.splits(R)
if SMOKE:
    tr, va, te = tr[:64], va[:32], {k: v[:16] for k, v in te.items()}
ALL = tr + va + te[1] + te[2] + te[3] + te[4]          # fixed order => cache is reproducible
print(f"{MODEL}: train={len(tr)} val={len(va)} tests={[len(te[k]) for k in range(1, 5)]} "
      f"| rows with image: {sum(1 for i in ALL if R[i]['img'])}/{len(ALL)}", flush=True)
bs = 4 if MODEL.startswith("qwen") else 32
X = np.concatenate([build_cache(f, R, ALL, bs) for f in NEEDS], axis=1).astype(np.float32)
pos = {i: n for n, i in enumerate(ALL)}
def XY(ids): return torch.tensor(X[[pos[i] for i in ids]]), [R[i]["y"] for i in ids]


class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(d, 384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 1))
    def forward(s, x): return torch.sigmoid(s.m(x).squeeze(1))


Xtr, ytr = XY(tr); Xva, yva = XY(va)
head = MLP(X.shape[1]); opt = torch.optim.Adam(head.parameters(), lr=2e-5, weight_decay=5e-5); bce = nn.BCELoss()
yt = torch.tensor(ytr, dtype=torch.float)
def pr(Xs):
    head.eval()
    with torch.no_grad(): return head(Xs).numpy().tolist()
best, bad, bs_state = -1, 0, None
for ep in range(1 if SMOKE else 50):
    head.train(); order = list(range(len(tr))); random.Random(C.SEED + ep).shuffle(order)
    for i in range(0, len(order), 64):
        b = order[i:i + 64]; loss = bce(head(Xtr[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    vm = C.metr(yva, pr(Xva), THR)["metric"]; print(f"ep{ep} val_macroF1@{THR}={vm:.4f}", flush=True)
    if vm > best: best, bad, bs_state = vm, 0, {k: v.clone() for k, v in head.state_dict().items()}
    else:
        bad += 1
        if bad >= 5: print("early stop", flush=True); break
head.load_state_dict(bs_state)
C.report(MODEL + ("_SMOKE" if SMOKE else ""), THR, R, te, {k: pr(XY(te[k])[0]) for k in range(1, 5)})
print("ALL DONE", flush=True)
