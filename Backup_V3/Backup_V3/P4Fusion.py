"""
PERSON 4 -- fusion + open-set decision.   python P4Fusion.py [--NewBias 0.5]
Needs Work/Folds.pkl, Work/Cands*.pkl (P2); uses Work/SiftPairs.pkl + Nkp.pkl (P3) if present.
  1) identity model: P(candidate identity is correct)   -- LightGBM, out-of-fold
  2) new model:      P(query is new_individual)          -- LightGBM on query-level features
  3) stage 2: same two models again, plus evidence from neighbouring queries
     (another test image of the same animal that confidently matches identity X)
  4) rank identities AND new_individual by probability (optimal for MAP@5) -> SubmissionFusion.csv
Allowed methods used: gradient boosting (LightGBM), k-NN / distance-based retrieval.
TODO:
  [ ] feature importance -> drop useless ones, add P3's extra features
  [ ] isotonic calibration before mixing p and pNew
  [ ] use --NewBias + public LB to match the test set's new_individual rate
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from Common import (NEW, WORK, LoadMeta, IdMaps, LoadFolds, LoadPkl, Map5, Side,
                    ViewRelation, WriteSubmission)
import Stage2

PARAMS = dict(n_estimators=800, learning_rate=0.015, num_leaves=15, min_child_samples=80,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
ID_FEATS = ["CosMax", "Cos2nd", "Cos3", "CosMean3", "Knn10", "SameSideMax", "SameGap", "IdRank",
            "QTop1", "QGap12", "CosGap", "LogNGal",
            "InlMax", "GoodMax", "InlSum", "InlCnt8", "NChecked", "InlRank", "QInlMax", "InlFrac",
            "InlNorm", "InlMargin",
            "QN15", "LogNkpQ", "VRel", "SpeciesC", "QSide"]
Q_BASE = ["QTop1", "QGap12", "QInlMax", "QInl2nd", "QN15", "QKnnTop", "LogNkpQ", "SpeciesC", "QSide"]
Q_FEATS = Q_BASE + ["PMax", "P2nd", "PSum"]
# stage 2: evidence from neighbouring queries (other test images of the same animal)
ID_FEATS2 = ID_FEATS + ["NbrP", "NbrPInl", "NbrCos", "NbrInl"]
Q_BASE2 = Q_BASE + ["NbrCos", "NbrInl", "NbrNew", "NbrPMax"]
Q_FEATS2 = Q_BASE2 + ["PMax", "P2nd", "PSum"]


def Second(s):
    s = s.dropna()
    return s.nlargest(2).iloc[-1] if len(s) >= 2 else np.nan


def AddFeatures(c, sift, nkp, vp, sp):
    c = c.copy()
    im, gm, isum, nkg, n8 = [], [], [], [], []
    for q, imgs in zip(c["Query"], c["BestImgs"]):
        v = [sift[(q, g)] for g in imgs if (q, g) in sift]
        n8.append(sum(x[1] >= 8 for x in v) if v else np.nan)
        im.append(max(x[1] for x in v) if v else np.nan)
        gm.append(max(x[0] for x in v) if v else np.nan)
        isum.append(sum(x[1] for x in v) if v else np.nan)
        nkg.append(nkp.get(imgs[0], np.nan))
    c["InlMax"], c["GoodMax"], c["InlSum"], c["InlCnt8"] = im, gm, isum, n8
    c["NChecked"] = c["BestImgs"].str.len()
    g = c.groupby("Query")["InlMax"]
    c["QInlMax"] = g.transform("max")
    c["QInl2nd"] = g.transform(Second)
    c["InlRank"] = g.rank(ascending=False, method="min")
    c["InlFrac"] = c.InlMax / (c.QInlMax + 1)
    nkq = c["Query"].map(nkp).astype(float)
    c["InlNorm"] = c.InlMax / (np.minimum(nkq, np.array(nkg, float)) + 1)
    c["LogNkpQ"] = np.log1p(nkq)
    c["LogNGal"] = np.log1p(c.NGal)
    c["CosGap"] = c.QTop1 - c.CosMax
    c["SameGap"] = c.CosMax - c.SameSideMax
    # inlier margin over the best OTHER candidate identity of the same query
    c["InlMargin"] = c.InlMax - np.where(c.InlMax >= c.QInlMax, c.QInl2nd, c.QInlMax)
    c["QN15"] = (c.InlMax >= 15).groupby(c["Query"]).transform("sum")
    c["QKnnTop"] = c.groupby("Query")["Knn10"].transform("max")
    c["VRel"] = [ViewRelation(vp[q], vp[imgs[0]]) for q, imgs in zip(c["Query"], c["BestImgs"])]
    c["SpeciesC"] = (c["Query"].map(sp) == "leopard").astype(int)
    c["QSide"] = c["Query"].map(lambda q: "LROU".index(Side(vp[q])))
    return c


def Oof(X, y, groups, n=5):
    out = np.zeros(len(X))
    for tr, va in GroupKFold(n).split(X, y, groups):
        m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        out[va] = m.predict_proba(X.iloc[va])[:, 1]
    return out


def QueryTable(c, base=Q_BASE):
    g = c.groupby("QKey")
    Q = g[base].first()
    Q["PMax"] = g.P.max()
    Q["P2nd"] = g.P.apply(Second).fillna(0)
    Q["PSum"] = g.P.sum()
    return Q


def Prepare(c):
    return [(qk, g.Identity.tolist(), g.P.to_numpy()) for qk, g in c.groupby("QKey", sort=False)]


def FinalPreds(prep, pNew, bias=0.0, normalise=True):
    """Score every label by probability (new_individual included) and keep the top 5."""
    preds = {}
    for qk, labs, pk in prep:
        p0 = np.clip(pNew[qk], 1e-6, 1 - 1e-6)
        pn = 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + bias)))
        sk = (1 - pn) * (pk / max(pk.sum(), 1.0) if normalise else pk)
        scores = np.r_[sk, pn]
        allL = labs + [NEW]
        preds[qk] = [allL[i] for i in np.argsort(-scores)[:5]]
    return preds


def FitStage(V, idFeats, qBase, qFeats, lab):
    """Out-of-fold identity probabilities + P(new) for the validation table V."""
    y = (V.Identity == V.Truth).to_numpy(int)
    V["P"] = Oof(V[idFeats], y, V["Query"].map(lab))
    QV = QueryTable(V, qBase)
    yNew = (V.groupby("QKey").Truth.first().loc[QV.index] == NEW).to_numpy(int)
    pNew = pd.Series(Oof(QV[qFeats], yNew, QV.index.str.split(":").str[1].map(lab)), index=QV.index)
    return y, QV, yNew, pNew


def WeightedMap5(truths, preds, newRate=None):
    """MAP@5 over all folds; if newRate is given, new_individual queries are re-weighted so
    they make up that share (to mimic a test set with a different new-animal rate)."""
    t = {k: v for tr in truths for k, v in tr.items()}
    s = np.array([next((1 / (k + 1) for k, l in enumerate(preds.get(q, [])[:5]) if l == v), 0.0)
                  for q, v in t.items()])
    isNew = np.array([v == NEW for v in t.values()])
    if newRate is None:
        return s.mean()
    return newRate * s[isNew].mean() + (1 - newRate) * s[~isNew].mean()


def TuneMix(V, pNew, truths, newRate=None):
    prep = Prepare(V)
    best = (-1, None, None)
    for norm in (True, False):
        for bias in np.linspace(-3, 2, 21):
            P = FinalPreds(prep, pNew, bias, norm)
            best = max(best, (WeightedMap5(truths, P, newRate), norm, bias))
    return best


def EstimateNewRate(pTest, valRate):
    """Prior-shift EM (Saerens et al. 2002): the share of new animals in the test set that is
    consistent with the model's test probabilities, given it was trained at valRate."""
    p = valRate
    for _ in range(200):
        w = (p / valRate) / ((1 - p) / (1 - valRate))
        p = float(np.mean(pTest * w / (pTest * w + 1 - pTest)))
    return p


def AddNeighbours(V, pNew, sets, sift):
    """Stage-2 features, computed separately for each fold / the test set."""
    out = []
    for prefix, neigh in sets:
        part = V[V.QKey.str.startswith(prefix)] if prefix else V
        if len(part):
            part = Stage2.NeighbourFeatures(part, pNew.to_dict(), neigh, prefix, sift)
            part["NbrPMax"] = part.groupby("Query")["NbrP"].transform("max")
            out.append(part)
    return pd.concat(out).loc[V.index]


def Main(newBias=None, stage2=True, newRate="auto"):
    train, test = LoadMeta()
    sp, vp, lab = IdMaps(train, test)
    folds = LoadFolds()
    sift = LoadPkl("SiftPairs.pkl", default={})
    nkp = LoadPkl("Nkp.pkl", default={})
    if not sift:
        print("WARNING: no SIFT cache yet -> cosine-only fusion")
    truths = [{f"{f['fold']}:{q}": t for q, t in f["truth"].items()} for f in folds]

    parts = []
    for f in folds:
        c = AddFeatures(pd.read_pickle(WORK / f"CandsFold{f['fold']}.pkl"), sift, nkp, vp, sp)
        c["QKey"] = f"{f['fold']}:" + c["Query"]
        c["Truth"] = c["Query"].map(f["truth"])
        parts.append(c)
    V = pd.concat(parts, ignore_index=True)
    ct = AddFeatures(pd.read_pickle(WORK / "CandsTest.pkl"), sift, nkp, vp, sp)
    ct["QKey"] = ct["Query"]

    # ---- stage 1 ----
    y, QV, yNew, pNew = FitStage(V, ID_FEATS, Q_BASE, Q_FEATS, lab)
    m, norm, bias = TuneMix(V, pNew, truths)
    print(f"stage 1 CV MAP@5 {m:.4f}  (normalise={norm}, bias={bias:+.1f})", flush=True)
    mId = lgb.LGBMClassifier(**PARAMS).fit(V[ID_FEATS], y)
    ct["P"] = mId.predict_proba(ct[ID_FEATS])[:, 1]
    QT = QueryTable(ct)
    mNew = lgb.LGBMClassifier(**PARAMS).fit(QV[Q_FEATS], yNew)
    pNewT = pd.Series(mNew.predict_proba(QT[Q_FEATS])[:, 1], index=QT.index)
    idF, qB, qF = ID_FEATS, Q_BASE, Q_FEATS

    # ---- stage 2: neighbouring queries ----
    if stage2:
        sets = Stage2.AllNeighbourSets(folds, test, sp)
        V = AddNeighbours(V, pNew, sets[:-1], sift)
        ct = AddNeighbours(ct, pNewT, sets[-1:], sift)
        y, QV, yNew, pNew = FitStage(V, ID_FEATS2, Q_BASE2, Q_FEATS2, lab)
        m2, norm2, bias2 = TuneMix(V, pNew, truths)
        print(f"stage 2 CV MAP@5 {m2:.4f}  (normalise={norm2}, bias={bias2:+.1f})", flush=True)
        if m2 > m:
            m, norm, bias = m2, norm2, bias2
            mId = lgb.LGBMClassifier(**PARAMS).fit(V[ID_FEATS2], y)
            ct["P"] = mId.predict_proba(ct[ID_FEATS2])[:, 1]
            QT = QueryTable(ct, Q_BASE2)
            mNew = lgb.LGBMClassifier(**PARAMS).fit(QV[Q_FEATS2], yNew)
            pNewT = pd.Series(mNew.predict_proba(QT[Q_FEATS2])[:, 1], index=QT.index)
            idF = ID_FEATS2
        else:
            print("stage 2 did not help -> keeping stage 1")

    print(f"final CV MAP@5 {m:.4f}  (at the validation new-animal rate {yNew.mean():.1%})")
    # ---- match the test set's share of new animals ----
    if newRate == "auto":
        newRate = EstimateNewRate(pNewT.to_numpy(), yNew.mean())
        print(f"estimated new-animal share in test: {newRate:.1%}")
    if newRate is not None:
        mT, norm, bias = TuneMix(V, pNew, truths, newRate)
        print(f"CV MAP@5 re-weighted to {newRate:.1%} new: {mT:.4f}  (normalise={norm}, bias={bias:+.2f})")
    if newBias is not None:
        bias = newBias
        print("using --NewBias", bias)
    print("top identity-model features:\n",
          pd.Series(mId.feature_importances_, idF).sort_values(ascending=False).head(10).to_string())
    print(f"predicted new_individual rate on test: {(pNewT > 0.5).mean():.1%}")
    WriteSubmission(FinalPreds(Prepare(ct), pNewT, bias, norm), test, train, "SubmissionFusion.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--NewBias", type=float, default=None)
    ap.add_argument("--NoStage2", action="store_true")
    ap.add_argument("--NewRate", default="auto",
                    help="share of new animals in test: 'auto' (estimated), a number such as 0.24, or 'none'")
    a = ap.parse_args()
    rate = None if a.NewRate == "none" else ("auto" if a.NewRate == "auto" else float(a.NewRate))
    Main(a.NewBias, not a.NoStage2, rate)
