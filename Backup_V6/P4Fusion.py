"""
PERSON 4 -- fusion + open-set decision.   python P4Fusion.py [--NewBias 0.5]
Needs Work/Folds.pkl, Work/Cands*.pkl (P2); uses Work/SiftPairs.pkl + Nkp.pkl (P3) if present.
  1) identity model: P(candidate identity is correct)   -- LightGBM, out-of-fold
  2) new model:      P(query is new_individual)          -- LightGBM on query-level features
  3) stage 2: same two models again, plus evidence from neighbouring queries
     (another test image of the same animal that confidently matches identity X)
  4) rank identities AND new_individual by probability (optimal for MAP@5)
  5) V6: a LightGBM RANKER (lambdarank) sees each query's candidates plus one new_individual row and
     learns the whole ranking directly; its scores are blended with steps 1-4 (simple model
     combination), weights tuned out-of-fold -> SubmissionFusion.csv
     (python P4Fusion.py --NoRanker reproduces V5)
Feature history: V4 added multi-space similarity + side views; V5 added comparable-view SIFT evidence;
V6 added the ranker blend.
Allowed methods used: gradient boosting (LightGBM, incl. lambdarank), k-NN / distance-based retrieval,
simple model combination.
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
            "QN15", "LogNkpQ", "VRel", "SpeciesC", "QSide",
            # engineered (V4): the same comparison seen in other projections of the MiewID vectors,
            # and evidence from the animal's other-side / front-back photos
            "RawCosBest", "RawCosMax", "RawRank", "WCosMax", "WRank", "RawGap", "WGap",
            "OppSideMax", "OtherViewMax", "NSides", "HasSameSide", "OppGap",
            # V5: was a SIFT comparison against a COMPARABLE view (same flank / same front-back label)?
            # separates "matched and failed" from "never had a comparable photo to match"
            "NCompChecked", "CompInlMax", "CompInlFrac", "NonCompInlMax", "NoCompView", "LogNCompGal"]
Q_BASE = ["QTop1", "QGap12", "QInlMax", "QInl2nd", "QN15", "QKnnTop", "LogNkpQ", "SpeciesC", "QSide",
          "QRawTop1", "QWTop1", "QCompInlMax"]
Q_FEATS = Q_BASE + ["PMax", "P2nd", "PSum"]
# stage 2: evidence from neighbouring queries (other test images of the same animal)
ID_FEATS2 = ID_FEATS + ["NbrP", "NbrPInl", "NbrCos", "NbrInl"]
Q_BASE2 = Q_BASE + ["NbrCos", "NbrInl", "NbrNew", "NbrPMax"]
Q_FEATS2 = Q_BASE2 + ["PMax", "P2nd", "PSum"]
RANK_PARAMS = dict(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
                   min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   lambdarank_truncation_level=10, verbose=-1, random_state=0)


def Second(s):
    s = s.dropna()
    return s.nlargest(2).iloc[-1] if len(s) >= 2 else np.nan


_RAW = {}


def RawUnit():
    """L2-normalised raw MiewID vectors (loaded once)."""
    if not _RAW:
        from Common import LoadMiewid
        F, row = LoadMiewid()
        _RAW["FN"], _RAW["row"] = F / np.linalg.norm(F, axis=1, keepdims=True), row
    return _RAW["FN"], _RAW["row"]


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
    # engineered features (V4)
    FN, row = RawUnit()
    q = c["Query"].map(row).to_numpy()
    b = c["BestImgs"].str[0].map(row).to_numpy()
    c["RawCosBest"] = (FN[q] * FN[b]).sum(1)
    c["HasSameSide"] = c.SameSideMax.notna().astype(int)
    c["OppGap"] = c.CosMax - c.OppSideMax
    c["QRawTop1"] = c.groupby("Query").RawCosMax.transform("max")
    c["QWTop1"] = c.groupby("Query").WCosMax.transform("max")
    c["RawGap"] = c.QRawTop1 - c.RawCosMax
    c["WGap"] = c.QWTop1 - c.WCosMax
    # V5: comparable-view SIFT evidence
    nc, ci, cf, ni = [], [], [], []
    for q, imgs in zip(c["Query"], c["BestImgs"]):
        qs = Side(vp[q])
        comp, other = [], []
        for g in imgs:
            v = sift.get((q, g))
            if v is None:
                continue
            same = (Side(vp[g]) == qs) if qs in "LR" else (vp[g] == vp[q] if qs == "O" else False)
            (comp if same else other).append((v[1], v[1] / (min(nkp.get(q, 0), nkp.get(g, 0)) + 1)))
        nc.append(len(comp) if qs != "U" else np.nan)
        ci.append(max((x[0] for x in comp), default=np.nan))
        cf.append(max((x[1] for x in comp), default=np.nan))
        ni.append(max((x[0] for x in other), default=np.nan))
    c["NCompChecked"], c["CompInlMax"], c["CompInlFrac"], c["NonCompInlMax"] = nc, ci, cf, ni
    c["NoCompView"] = np.where(c.NCompGal < 0, np.nan, (c.NCompGal == 0).astype(float))
    c["LogNCompGal"] = np.log1p(c.NCompGal.clip(lower=0))
    c["QCompInlMax"] = c.groupby("Query").CompInlMax.transform("max")
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


# ---------------------------------------------------------------- V6: ranker + blend
def RankRows(c, qCols):
    """Each query's candidate rows plus one new_individual row (query-level features copied)."""
    R = c.copy()
    R["IsNewRow"] = 0
    q = c.groupby("QKey", sort=False).first().reset_index()
    n = pd.DataFrame({"QKey": q.QKey, "Query": q.Query, "Identity": NEW, "IsNewRow": 1})
    if "Truth" in q:
        n["Truth"] = q.Truth.values
    for col in qCols:
        n[col] = q[col].values
    R = pd.concat([R, n], ignore_index=True)
    return R.sort_values(["QKey", "IsNewRow"], kind="stable").reset_index(drop=True)


def RankFeats(idF, qB):
    return list(dict.fromkeys(idF + [c for c in qB if c not in idF] + ["IsNewRow"]))


def OofRank(R, feats, lab):
    """Out-of-fold ranker scores; grouped by animal so a query's own animal never trains its scorer."""
    y = (R.Identity == R.Truth).to_numpy(int)
    out = np.zeros(len(R))
    for tr, va in GroupKFold(5).split(R, y, R["Query"].map(lab)):
        Rt = R.iloc[tr]
        m = lgb.LGBMRanker(**RANK_PARAMS).fit(Rt[feats], y[tr], group=Rt.groupby("QKey", sort=False).size().to_numpy())
        out[va] = m.predict(R.iloc[va][feats])
    return out


def _GroupSum(x, code):
    return pd.Series(x).groupby(code).transform("sum").to_numpy()


def BlendScores(R, pNewMap, rankScore, a, b1, b2, norm=True):
    """a * (V5 probabilities, normalised per query) + (1 - a) * (ranker softmax per query).
    b1 / b2 shift new_individual in the V5 part / the ranker part."""
    code = pd.factorize(R.QKey)[0]
    isNew = R.IsNewRow.to_numpy() == 1
    p = np.nan_to_num(R.P.to_numpy())
    pn0 = np.clip(R.QKey.map(pNewMap).to_numpy(), 1e-6, 1 - 1e-6)
    pn = 1 / (1 + np.exp(-(np.log(pn0 / (1 - pn0)) + b1)))
    psum = _GroupSum(p * ~isNew, code)
    v5 = np.where(isNew, pn, (1 - pn) * (p / np.maximum(psum, 1.0) if norm else p))
    v5 = v5 / _GroupSum(v5, code)
    s = rankScore + b2 * isNew
    s = s - pd.Series(s).groupby(code).transform("max").to_numpy()
    e = np.exp(s)
    return a * v5 + (1 - a) * e / _GroupSum(e, code)


def TopFive(R, score):
    code = pd.factorize(R.QKey)[0]
    order = np.lexsort((-score, code))
    g = code[order]
    pos = np.arange(len(g)) - np.searchsorted(g, g)
    keep = order[pos < 5]
    top = R.iloc[keep][["QKey", "Identity"]]
    return top.groupby("QKey", sort=False).Identity.apply(list).to_dict()


def BlendMap(R, score, newRate):
    """MAP@5 of blended scores; new_individual queries re-weighted to newRate."""
    code, keys = pd.factorize(R.QKey)
    order = np.lexsort((-score, code))
    g = code[order]
    pos = np.arange(len(g)) - np.searchsorted(g, g)
    hit = (R.Identity.to_numpy() == R.Truth.to_numpy())[order] & (pos < 5)
    per = np.zeros(len(keys))
    per[g[hit]] = 1.0 / (pos[hit] + 1)
    isNewQ = R.groupby("QKey", sort=False).Truth.first().reindex(keys).to_numpy() == NEW
    if newRate is None:
        return per.mean()
    return newRate * per[isNewQ].mean() + (1 - newRate) * per[~isNewQ].mean()


def Main(newBias=None, stage2=True, newRate="auto", ranker=True):
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
            idF, qB = ID_FEATS2, Q_BASE2
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
    if not ranker:
        WriteSubmission(FinalPreds(Prepare(ct), pNewT, bias, norm), test, train, "SubmissionFusion.csv")
        return

    # ---- V6: lambdarank ranker, blended with the models above ----
    feats = RankFeats(idF, qB)
    R, RT = RankRows(V, qB), RankRows(ct, qB)
    print("training the ranker out-of-fold ...", flush=True)
    rOof = OofRank(R, feats, lab)
    rate = newRate if newRate is not None else yNew.mean()
    best = (-1, None, None, None)
    for a in (0.2, 0.3, 0.4):
        for b1 in np.linspace(-1.5, 1.0, 6):
            for b2 in np.linspace(-2.0, 0.5, 6):
                mB = BlendMap(R, BlendScores(R, pNew, rOof, a, b1, b2, norm), rate)
                best = max(best, (mB, a, b1, b2))
    mB, a, b1, b2 = best
    print(f"V6 blend CV MAP@5 re-weighted to {rate:.1%} new: {mB:.4f}  "
          f"(weight on steps 1-4 {a}, their new-bias {b1:+.2f}, ranker new-bias {b2:+.2f})")
    y = (R.Identity == R.Truth).to_numpy(int)
    mR = lgb.LGBMRanker(**RANK_PARAMS).fit(R[feats], y, group=R.groupby("QKey", sort=False).size().to_numpy())
    preds = TopFive(RT, BlendScores(RT, pNewT, mR.predict(RT[feats]), a, b1, b2, norm))
    WriteSubmission(preds, test, train, "SubmissionFusion.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--NewBias", type=float, default=None)
    ap.add_argument("--NoStage2", action="store_true")
    ap.add_argument("--NoRanker", action="store_true", help="skip the V6 ranker blend (reproduces V5)")
    ap.add_argument("--NewRate", default="auto",
                    help="share of new animals in test: 'auto' (estimated), a number such as 0.24, or 'none'")
    a = ap.parse_args()
    rate = None if a.NewRate == "none" else ("auto" if a.NewRate == "auto" else float(a.NewRate))
    Main(a.NewBias, not a.NoStage2, rate, not a.NoRanker)
