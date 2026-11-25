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

PARAMS = dict(n_estimators=800, learning_rate=0.015, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
ID_FEATS = ["CosMax", "Cos2nd", "Cos3", "CosMean3", "Knn10", "SameSideMax", "SameGap", "IdRank",
            "QTop1", "QGap12", "CosGap", "LogNGal",
            "InlMax", "GoodMax", "InlSum", "InlRank", "QInlMax", "InlFrac", "InlNorm", "InlMargin",
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
    im, gm, isum, nkg = [], [], [], []
    for q, imgs in zip(c["Query"], c["BestImgs"]):
        v = [sift[(q, g)] for g in imgs if (q, g) in sift]
        im.append(max(x[1] for x in v) if v else np.nan)
        gm.append(max(x[0] for x in v) if v else np.nan)
        isum.append(sum(x[1] for x in v) if v else np.nan)
        nkg.append(nkp.get(imgs[0], np.nan))
    c["InlMax"], c["GoodMax"], c["InlSum"] = im, gm, isum
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

