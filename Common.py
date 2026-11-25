"""
Common.py -- shared helpers for the whole team (OWNER: Person 1).
Keep function names stable: every other script imports from here.

projectwild3/
    Common.py  P1Validation.py  P2Global.py  P3Sift.py  P4Fusion.py
    reid_student_data_main/   (provided, read-only)
    sift_features_train.npz/  (provided, read-only; folder of .npy OR a real .npz)
    sift_features_test.npz/   (provided, read-only)
    train_images_hyena/ train_images_leopard/   (provided, NEVER read -- competition rule)
    Work/                     created automatically; all hand-off files go here

Data elsewhere?  Windows: set REID_BASE=C:\\path\\to\\projectwild3
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("REID_BASE", Path(__file__).resolve().parent))
DATA = BASE / "reid_student_data_main"
SIFT_SRC = {"tr": BASE / "sift_features_train.npz", "te": BASE / "sift_features_test.npz"}
WORK = BASE / "Work"
WORK.mkdir(exist_ok=True)
NEW = "new_individual"


# ---------------- metadata ----------------
def LoadMeta():
    """(train, test) DataFrames; missing viewpoint -> 'unknown'."""
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for d in (train, test):
        d["viewpoint"] = d["viewpoint"].fillna("unknown")
    return train, test


def IdMaps(train, test):
    """id->species, id->viewpoint, train id->identity"""
    a = pd.concat([train, test], ignore_index=True)
    return dict(zip(a.id, a.species)), dict(zip(a.id, a.viewpoint)), dict(zip(train.id, train.identity))


SIDES = {"left": "L", "frontleft": "L", "backleft": "L",
         "right": "R", "frontright": "R", "backright": "R"}


def Side(vp):
    """L / R flank, O other (front/back/up/down), U unknown"""
    return "U" if vp == "unknown" else SIDES.get(vp, "O")


def ViewRelation(a, b):
    """0 same side, 1 different, 2 opposite flanks (L vs R), 3 unknown"""
    a, b = Side(a), Side(b)
    if "U" in (a, b):
        return 3
    if a == b:
        return 0
    return 2 if {a, b} == {"L", "R"} else 1


# ---------------- features ----------------
def LoadMiewid():
    """(F, Row): F is (9954, 2152) float32, Row[id] -> row index"""
    F = np.load(DATA / "miewid_features.npy").astype(np.float32)
    ids = pd.read_csv(DATA / "miewid_ids.csv")["id"].tolist()
    assert len(ids) == len(F), "miewid ids and features do not line up"
    return F, {i: k for k, i in enumerate(ids)}


EMPTY = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
NPZ = {}


def LoadSift(imgId):
    """(xy (n,2) float32, desc (n,128) uint8); empty arrays if no keypoints.
    Works with an extracted folder of .npy files or the original .npz file.
    Not thread-safe for the .npz case -- load in the main thread."""
    src = SIFT_SRC[imgId[:2]]
    if src.is_dir():
        fx, fd = src / f"{imgId}_xy.npy", src / f"{imgId}_desc.npy"
        if not (fx.exists() and fd.exists()):
            return EMPTY
        xy, d = np.load(fx), np.load(fd)
    else:
        if src not in NPZ:
            NPZ[src] = np.load(src)
        z = NPZ[src]
        if f"{imgId}_xy" not in z.files or f"{imgId}_desc" not in z.files:
            return EMPTY
        xy, d = z[f"{imgId}_xy"], z[f"{imgId}_desc"]
    if len(d) == 0:
        return EMPTY
    return xy.astype(np.float32).reshape(-1, 2), d.astype(np.uint8).reshape(-1, 128)


# ---------------- metric ----------------
def Map5(truth, preds):
    """truth {qid: label}, preds {qid: [up to 5 labels]} -> MAP@5"""
    total = 0.0
    for q, t in truth.items():
        for k, lab in enumerate(preds.get(q, [])[:5]):
            if lab == t:
                total += 1.0 / (k + 1)
                break
    return total / max(len(truth), 1)


# ---------------- validation folds ----------------
def MakeFolds(train, nFolds=4, unkFrac=0.2, qFrac=0.25, cap=10, seed=42):
    """Simulate the test set from train. Per fold: hide ~unkFrac of identities entirely
    (their queries' truth = new_individual) and use ~qFrac of the other identities'
    images as queries. Returns [{fold, queryIds, galleryIds, truth}]"""
    rng = np.random.default_rng(seed)
    lab = dict(zip(train.id, train.identity))
    folds = []
    for f in range(nFolds):
        queries, hidden = [], set()
        for _, d in train.groupby("species"):
            idents = rng.permutation(np.asarray(d["identity"].unique(), dtype=object))
            unk = set(idents[: int(round(unkFrac * len(idents)))])
            for ident, g in d.groupby("identity"):
                ids = g["id"].to_numpy()
                if ident in unk:
                    hidden.update(ids)
                    queries += list(rng.choice(ids, min(cap, len(ids)), replace=False))
                elif len(ids) >= 2:
                    k = min(cap, max(1, int(round(qFrac * len(ids)))), len(ids) - 1)
                    queries += list(rng.choice(ids, k, replace=False))
        qset = set(queries)
        gallery = [i for i in train.id if i not in qset and i not in hidden]
        galIdents = {lab[i] for i in gallery}
        truth = {q: (lab[q] if lab[q] in galIdents else NEW) for q in sorted(qset)}
        folds.append({"fold": f, "queryIds": sorted(qset), "galleryIds": gallery, "truth": truth})
    return folds


def SavePkl(obj, name):
    with open(WORK / name, "wb") as fh:
        pickle.dump(obj, fh)


def LoadPkl(name, default=None):
    p = WORK / name
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"{p} missing -- has that step been run?")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def LoadFolds():
    return LoadPkl("Folds.pkl")


# ---------------- submission ----------------
def CheckSubmission(df, train, test):
    """AssertionError if Kaggle would reject the file or a label can never score."""
    spIdent = dict(zip(train.identity, train.species))
    spTest = dict(zip(test.id, test.species))
    assert list(df.columns) == ["id", "predictions"], "columns must be id,predictions"
    assert len(df) == len(test) and set(df.id) == set(test.id), "need one row per test id"
    assert not df.id.duplicated().any(), "duplicate ids"
    for i, p in zip(df.id, df.predictions):
        labs = p.split()
        assert 1 <= len(labs) <= 5 and len(set(labs)) == len(labs), f"{i}: bad label list"
        for l in labs:
            assert l == NEW or spIdent.get(l) == spTest[i], f"{i}: bad label {l}"
    return True


def WriteSubmission(preds, test, train, path):
    """preds {test_id: [labels]} -> checked CSV in the project folder"""
    rows = []
    for i in test.id:
        labs = []
        for l in preds.get(i, []):
            if l not in labs:
                labs.append(l)
        rows.append((i, " ".join(labs[:5] or [NEW])))
    df = pd.DataFrame(rows, columns=["id", "predictions"])
    CheckSubmission(df, train, test)
    df.to_csv(BASE / path, index=False)
    print(f"wrote {path} ({len(df)} rows, format OK)")
    return df
