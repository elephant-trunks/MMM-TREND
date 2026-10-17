"""Shared data/protocol for the benchmark baselines on data/complete_dataset_v1.2.xlsx.
Protocol (same as the BERT runs): train once on rows dated <= 2023-Q4 (10% random val,
seed 2378), evaluate separately on each 2024 quarter + Avg.
Text  = full article body (data/articles_text/<File>), fallback Title + '. ' + native Summary.
Image = first resolvable of Best Match (CLIP), Best Match (BLIP), Image 1..10 ID in
        images/; else None
        (repo dataloader feeds a black 224x224 image in that case).
Labels: True/Real->real(0); False/Fake/Problematic->fake(1); Maybe->per-quarter minority."""
import openpyxl, re, os, json, collections, random, numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# repository root (this file lives in code/benchmark/); outputs are written to ROOT/bench_*
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLSX = os.path.join(ROOT, "data", "complete_dataset_v1.2.xlsx")
ART = os.path.join(ROOT, "data", "articles_text")
AUG = ART
IMG_DIRS = [os.path.join(ROOT, "images")]
SEED = 2378
IMG_COLS = ["Best Match Image (CLIP)", "Best Match Image (BLIP)"] + [f"Image {i} ID" for i in range(1, 11)]
YEARMAP = {str(y): i for i, y in enumerate(range(2012, 2025))}


def cell(v):
    return "" if v is None else str(v).strip()


def qof(v):
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", cell(v))
    return (int(m[3]), (int(m[2]) - 1) // 3 + 1) if m else None


def _body(f):
    f = cell(f)
    if not f:
        return ""
    p = os.path.join(AUG if f.startswith("ID-AUG-") else ART, f)
    if os.path.exists(p):
        try:
            return open(p, encoding="utf-8", errors="ignore").read().strip()
        except Exception:
            return ""
    return ""


def load_rows():
    have = [set(os.listdir(d)) if os.path.isdir(d) else set() for d in IMG_DIRS]
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    ws = wb["Complete-Dataset"]
    it = ws.iter_rows(values_only=True)
    H = [h for h in next(it) if h is not None]
    idx = {h: i for i, h in enumerate(H)}
    R = []
    for r in it:
        q = qof(r[idx["dates"]])
        if not q or not (2012 <= q[0] <= 2024):
            continue
        b = _body(r[idx["File"]])
        txt = b if (b and len(b) >= 20) else (cell(r[idx["Title"]]) + ". " + cell(r[idx["Summary"]])).strip()
        img = None
        for k in IMG_COLS:
            v = cell(r[idx[k]])
            if not v:
                continue
            for d, h in zip(IMG_DIRS, have):
                if v in h:
                    img = os.path.join(d, v)
                    break
            if img:
                break
        mm = re.match(r"(\d{2})/(\d{2})/(\d{4})", cell(r[idx["dates"]]))
        R.append({"text": txt, "auth_raw": cell(r[idx["Authenticity"]]), "yq": q, "month": int(mm[2]),
                  "year": str(q[0]), "img": img, "aug": cell(r[idx["File"]]).startswith("ID-AUG-"),
                  "region": cell(r[idx["Location_std"]]) if "Location_std" in idx else ""})
    wb.close()
    # label consolidation (paper rule)
    FIXED = {"True": "real", "Real": "real", "False": "fake", "Fake": "fake", "Problematic": "fake"}
    for x in R:
        x["lab"] = FIXED.get(x["auth_raw"])
    qc = collections.defaultdict(lambda: {"real": 0, "fake": 0})
    for x in R:
        if x["lab"]:
            qc[x["yq"]][x["lab"]] += 1
    for x in sorted([x for x in R if x["lab"] is None], key=lambda x: (x["yq"][0], x["yq"][1], x["auth_raw"])):
        c = qc[x["yq"]]
        cls = "real" if c["real"] < c["fake"] else "fake"
        c[cls] += 1
        x["lab"] = cls
    for x in R:
        x["y"] = 0 if x["lab"] == "real" else 1
    return R


def splits(R):
    """Returns index lists (into R): train, val, {1..4: test}. Identical to BERT runs."""
    cut = 2024 * 4 + 1
    tr = [i for i, x in enumerate(R) if x["yq"][0] * 4 + x["yq"][1] < cut]
    random.Random(SEED).shuffle(tr)
    nv = int(0.1 * len(tr))
    return tr[nv:], tr[:nv], {k: [i for i, x in enumerate(R) if x["yq"] == (2024, k)] for k in range(1, 5)}


def metr(y, pr, thr):
    yp = [1 if p > thr else 0 for p in pr]
    labs = sorted(set(y))
    f1v = f1_score(y, yp, average=None, zero_division=0)
    fr = f1v[labs.index(0)] if 0 in labs else 0.0
    fk = f1v[labs.index(1)] if 1 in labs else 0.0
    return {"metric": float(f1_score(y, yp, average="macro", zero_division=0)),
            "acc": float(accuracy_score(y, yp)), "f1_real": float(fr), "f1_fake": float(fk)}


def report(name, thr, R, tests, probs_by_q):
    """probs_by_q: {k: list of probs aligned with tests[k]}. Writes results + probs json."""
    res, store = {}, {}
    sweep_thr = [round(t, 2) for t in np.arange(0.20, 0.81, 0.05)]
    sweep = {}
    for k in range(1, 5):
        ids = tests[k]
        y = [R[i]["y"] for i in ids]
        p = probs_by_q[k]
        res[f"Q{k}"] = metr(y, p, thr)
        try:
            res[f"Q{k}"]["auc"] = float(roc_auc_score(y, p))
        except Exception:
            res[f"Q{k}"]["auc"] = None
        wi = [j for j, i in enumerate(ids) if R[i]["img"]]
        if wi and len(set(y[j] for j in wi)) > 1:
            res[f"Q{k}"]["with_image_only"] = dict(metr([y[j] for j in wi], [p[j] for j in wi], thr), n=len(wi))
        store[f"Q{k}"] = {"y": y, "p": p, "has_img": [bool(R[i]["img"]) for i in ids], "aug": [R[i]["aug"] for i in ids]}
        sweep[f"Q{k}"] = {t: metr(y, p, t)["metric"] for t in sweep_thr}
    res["Avg"] = {m: float(np.mean([res[f"Q{k}"][m] for k in range(1, 5)])) for m in ["metric", "acc", "f1_fake", "f1_real"]}
    aucs = [res[f"Q{k}"]["auc"] for k in range(1, 5) if res[f"Q{k}"]["auc"] is not None]
    res["Avg"]["auc"] = float(np.mean(aucs)) if aucs else None
    sweep["Avg"] = {t: float(np.mean([sweep[f"Q{k}"][t] for k in range(1, 5)])) for t in sweep_thr}
    out = {"model": name, "decision_boundary": thr, "protocol": "train-once <=2023-Q4, test each 2024 quarter",
           "results": res}
    os.makedirs(os.path.join(ROOT, "bench_results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "bench_results", f"{name}_results.json"), "w"), indent=2)
    json.dump({"threshold": thr, "probs": store, "sweep": sweep}, open(os.path.join(ROOT, "bench_results", f"{name}_probs.json"), "w"))
    print(f"\n===== {name} @ {thr} =====", flush=True)
    for k in ["Q1", "Q2", "Q3", "Q4", "Avg"]:
        r = res[k]
        print(f"{k}: MacroF1={r['metric']:.4f} Acc={r['acc']:.4f} F1fake={r['f1_fake']:.4f} F1real={r['f1_real']:.4f} AUC={r.get('auc')}", flush=True)
    return out


def parse_years(argv):
    """First arg like '2023' or '2023+2024' -> [2023] / [2023, 2024]; None if absent."""
    for a in argv[1:]:
        if re.fullmatch(r"\d{4}(\+\d{4})*", a):
            return [int(y) for y in a.split("+")]
    return None


def year_tag(years):
    return "_" + "_".join(map(str, years)) if years else ""


def year_split(R, years):
    """Train on all rows dated before the earliest test year (10% random val, seed 2378);
    test quarter k = rows of quarter k in any of the test years (pooled across years)."""
    years = [years] if isinstance(years, int) else list(years)
    tr = [i for i, x in enumerate(R) if x["yq"][0] < min(years)]
    random.Random(SEED).shuffle(tr)
    nv = int(0.1 * len(tr))
    return tr[nv:], tr[:nv], {k: [i for i, x in enumerate(R) if x["yq"][0] in years and x["yq"][1] == k] for k in range(1, 5)}


def save_trainable(state, name):
    """Save only the parts that train (BERT layer 11 + heads); frozen mBERT is re-loaded from the hub."""
    import torch
    keep = {k: v for k, v in state.items() if not k.startswith("bert.") or ".layer.11." in k}
    os.makedirs(os.path.join(ROOT, "bench_models"), exist_ok=True)
    torch.save(keep, os.path.join(ROOT, "bench_models", f"{name}.pt"))


# ---------------- spatial (leave-one-region-out) ----------------
REGIONS = {
    1: {"Tamil_Nadu", "Kerala", "Telangana_Andhra", "Karnataka"},                                   # R1 South
    2: {"West_Bengal", "Odisha", "Bihar", "Assam", "Northeast", "Jharkhand", "Bangladesh"},          # R2 East
    3: {"Gujarat", "Maharashtra", "Rajasthan", "Goa"},                                              # R3 West
    4: {"Delhi", "Uttar_Pradesh", "Jammu_Kashmir", "Punjab", "Madhya_Pradesh", "Uttarakhand",
        "Haryana", "Chhattisgarh", "Himachal_Pradesh"},                                             # R4 North/Central
}   # Pan_India and International are never tested -> always in training


def region_of(x):
    for k, s in REGIONS.items():
        if x["region"] in s:
            return k
    return 0


def region_split(R, k):
    """Fold k: train on every row not in region k (10% random val, seed 2378); test = region k."""
    tr = [i for i, x in enumerate(R) if region_of(x) != k]
    random.Random(SEED).shuffle(tr)
    nv = int(0.1 * len(tr))
    return tr[nv:], tr[:nv], [i for i, x in enumerate(R) if region_of(x) == k]
