"""
PERSON 2 -- MiewID retrieval + baseline submission.   python P2Global.py
Embedding: per-species centring -> PCA whitening (512) -> concat with an LDA projection (128)
           fitted ONLY on gallery labels (per fold; all train for test) -> cosine retrieval.
Writes Work/Emb{fold}.pkl, Work/EmbTest.pkl, Work/CandsFold{f}.pkl, Work/CandsTest.pkl,
       SubmissionBaseline.csv
Candidate table (one row per query x candidate identity) -- DON'T rename columns:
  Query Identity CosMax Cos2nd Cos3 CosMean3 Knn10 SameSideMax OppSideMax OtherViewMax NSides
  IdRank QTop1 QGap12 NGal BestImgs
  (OppSideMax: best cosine among the animal's photos of the OTHER flank; OtherViewMax: among its
   front/back/up/down photos; NSides: how many of L / R / other views the animal has in the gallery;
   RawCosMax / RawRank and WCosMax / WRank: best-photo cosine and identity rank in the raw MiewID
   space and in the whitened space without LDA -- the same comparison seen through other projections)
  BestImgs = top TOP_IMG gallery ids of that identity by cosine (+ best same-flank image);
             for the ALL_PHOTOS_TOP best identities: all their photos (up to ALL_PHOTOS_CAP)
Allowed methods used: PCA, LDA as a projection, distance-based retrieval / k-NN.
TODO:
  [ ] k-reciprocal re-ranking, alpha query expansion
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Common import (NEW, WORK, LoadMeta, IdMaps, LoadMiewid, LoadFolds, Map5, SavePkl,
                    WriteSubmission, Side)

TOP_ID, TOP_IMG = 50, 3
ALL_PHOTOS_TOP, ALL_PHOTOS_CAP = 3, 40  # for the 3 best candidates, SIFT-check (up to 40) of ALL their photos
N_PCA, N_LDA, LDA_MIX = 512, 128, 0.5
CAND_COLS = ["Query", "Identity", "CosMax", "Cos2nd", "Cos3", "CosMean3", "Knn10", "SameSideMax",
             "OppSideMax", "OtherViewMax", "NSides", "IdRank", "QTop1", "QGap12", "NGal", "BestImgs",
             "RawCosMax", "RawRank", "WCosMax", "WRank"]


def Whiten(F, row, sp, nComp=N_PCA):
    """Per-species centring + PCA whitening (unsupervised, fitted on train+test), then L2."""
    ids = sorted(row, key=row.get)
    spa = np.array([sp[i] for i in ids])
    E = np.zeros((len(F), nComp), np.float32)
    for s in np.unique(spa):
        m = spa == s
        Z = F[m].astype(np.float64)
        E[m] = PCA(nComp, whiten=True, random_state=0).fit_transform(Z - Z.mean(0))
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def AddLda(W, row, sp, lab, galleryIds, nComp=N_LDA, mix=LDA_MIX):
    """LDA projection fitted per species on GALLERY images only; cos = (1-mix)cosW + mix cosLDA."""
    ids = sorted(row, key=row.get)
    spa = np.array([sp[i] for i in ids])
    L = np.zeros((len(W), nComp), np.float32)
    for s in np.unique(spa):
        g = [i for i in galleryIds if sp[i] == s]
        y = np.array([lab[i] for i in g])
        k = min(nComp, len(np.unique(y)) - 1)
        m = LinearDiscriminantAnalysis(n_components=k, solver="eigen", shrinkage="auto")
        m.fit(W[[row[i] for i in g]], y)
        L[spa == s, :k] = m.transform(W[spa == s])
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-12
    return np.hstack([np.sqrt(1 - mix) * W, np.sqrt(mix) * L]).astype(np.float32)


def Retrieve(E, row, sp, vp, queryIds, gallery, extra=None):
    """gallery: DataFrame(id, identity, species) -> candidate table.
    extra: {"Raw": matrix, "W": matrix} other L2-normalised spaces scored for the same candidates."""
    extra = extra or {}
    out = []
    for s, G in gallery.groupby("species"):
        qids = [q for q in queryIds if sp[q] == s]
        if not qids:
            continue
        G = G.sort_values("identity")
        gid, glab = G.id.to_numpy(), G.identity.to_numpy()
        gside = np.array([Side(vp[g]) for g in gid])
        starts = np.r_[0, np.flatnonzero(glab[1:] != glab[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(glab)])
        seg = np.repeat(np.arange(len(starts)), counts)  # gallery position -> identity index
        labs = glab[starts]
        Ge = E[[row[g] for g in gid]]
        for b in range(0, len(qids), 256):
            qb = qids[b:b + 256]
            S = E[[row[q] for q in qb]] @ Ge.T
            M = np.maximum.reduceat(S, starts, axis=1)
            XM, XR = {}, {}
            for name, X in extra.items():  # per-identity best cosine + identity rank in other spaces
                XM[name] = np.maximum.reduceat(X[[row[q] for q in qb]] @ X[[row[g] for g in gid]].T, starts, axis=1)
                XR[name] = np.argsort(np.argsort(-XM[name], axis=1), axis=1)
            for r, q in enumerate(qb):
                knn = np.bincount(seg[np.argsort(-S[r])[:10]], minlength=len(starts))
                qs = Side(vp[q])
                order = np.argsort(-M[r])[:TOP_ID]
                top1 = M[r, order[0]]
                top2 = M[r, order[1]] if len(order) > 1 else -1.0
                for rk, j in enumerate(order):
                    a, n = starts[j], counts[j]
                    v = S[r, a:a + n]
                    o = np.argsort(-v)
                    best = list(gid[a + o[:(ALL_PHOTOS_CAP if rk < ALL_PHOTOS_TOP else TOP_IMG)]])
                    gs = gside[a:a + n]
                    same = np.flatnonzero(gs == qs) if qs in "LR" else np.array([], int)
                    ssm = float(v[same].max()) if len(same) else np.nan
                    opp = np.flatnonzero(gs == {"L": "R", "R": "L"}.get(qs, "-"))
                    oth = np.flatnonzero(gs == "O")
                    osm = float(v[opp].max()) if len(opp) else np.nan
                    ovm = float(v[oth].max()) if len(oth) else np.nan
                    nsides = len(set(gs) - {"U"})
                    if len(same):
                        bs = gid[a + same[np.argmax(v[same])]]
                        if bs not in best:
                            best.append(bs)
                    out.append((q, labs[j], float(v[o[0]]),
                                float(v[o[1]]) if n > 1 else np.nan,
                                float(v[o[2]]) if n > 2 else np.nan,
                                float(v[o[:3]].mean()), int(knn[j]), ssm, osm, ovm, nsides, rk,
                                float(top1), float(top1 - top2), int(n), best,
                                *(v for nm in ("Raw", "W") for v in
                                  ((float(XM[nm][r, j]), int(XR[nm][r, j])) if nm in XM else (np.nan, -1)))))
    return pd.DataFrame(out, columns=CAND_COLS)


def PerQuery(c):
    c = c.sort_values(["Query", "IdRank"])
    return {q: (g.Identity.tolist(), g.QTop1.iloc[0]) for q, g in c.groupby("Query", sort=False)}


def BaselinePreds(pq, t, pos):
    """new_individual first if top-1 cosine < t, otherwise at position `pos`."""
    preds = {}
    for q, (labs, top1) in pq.items():
        labs = labs[:5]
        labs.insert(0 if top1 < t else pos - 1, NEW)
        preds[q] = labs[:5]
    return preds


def Tune(pqs, truths):
    allT = np.concatenate([[v[1] for v in pq.values()] for pq in pqs])
    best = (-1, None, None)
    for t in np.quantile(allT, np.linspace(0, 0.8, 33)):
        for pos in (2, 3, 4, 5):
            m = np.mean([Map5(tr, BaselinePreds(pq, t, pos)) for pq, tr in zip(pqs, truths)])
            best = max(best, (m, t, pos))
    return best


def Main():
    train, test = LoadMeta()
    sp, vp, lab = IdMaps(train, test)
    F, row = LoadMiewid()
    folds = LoadFolds()
    tr = train[["id", "identity", "species"]]
    W = Whiten(F, row, sp)
    extra = {"Raw": F / np.linalg.norm(F, axis=1, keepdims=True), "W": W}
    cands = []
    for f in folds:
        E = AddLda(W, row, sp, lab, f["galleryIds"])
        SavePkl((E, row), f"Emb{f['fold']}.pkl")
        c = Retrieve(E, row, sp, vp, f["queryIds"], tr[tr.id.isin(set(f["galleryIds"]))], extra)
        c.to_pickle(WORK / f"CandsFold{f['fold']}.pkl")
        cands.append(c)
        print(f"fold {f['fold']} retrieved", flush=True)
    truths = [f["truth"] for f in folds]
    pqs = [PerQuery(c) for c in cands]
    rec = np.mean([np.mean([t[q] in pq[q][0] for q in t if t[q] != NEW]) for pq, t in zip(pqs, truths)])
    m, t, pos = Tune(pqs, truths)
    print(f"baseline CV MAP@5 {m:.4f} (t={t:.3f}, pos={pos})  recall@{TOP_ID} {rec:.3f}")
    E = AddLda(W, row, sp, lab, train.id.tolist())
    SavePkl((E, row), "EmbTest.pkl")
    ct = Retrieve(E, row, sp, vp, test.id.tolist(), tr, extra)
    ct.to_pickle(WORK / "CandsTest.pkl")
    WriteSubmission(BaselinePreds(PerQuery(ct), t, pos), test, train, "SubmissionBaseline.csv")


if __name__ == "__main__":
    Main()
