"""
PERSON 4 -- fusion + open-set decision.   python P4Fusion.py [--NewBias 0.5]
Needs Work/Folds.pkl, Work/Cands*.pkl (P2); uses Work/SiftPairs.pkl + Nkp.pkl (P3) if present.
  1) identity model: P(candidate identity is correct)   -- LightGBM, out-of-fold
  2) new model:      P(query is new_individual)          -- LightGBM on query-level features
  3) stage 2: same two models again, plus evidence from neighbouring queries
     (another test image of the same animal that confidently matches identity X)
  4) rank identities AND new_individual by probability (optimal for MAP@5)
  5) V8 ENSEMBLE (simple model combination of every allowed model family, same V5 features):
     classifier members (identity model + new-animal model, combined like step 4):
         lgb (= steps 1-4), cb (CatBoost), hgb (HistGradientBoosting), rf (random forest),
         lr (logistic regression), mlp (MLP, fully connected)
     ranker members (each query's candidates + one new_individual row, learn the ranking directly):
         lgbrank (LightGBM lambdarank, 3 seeds), xgbrank (XGBoost rank:map), cbrank (CatBoost YetiRank)
     Every member is trained out-of-fold; the blend weights and each member's new_individual shift are
     tuned on those out-of-fold scores -> SubmissionFusion.csv
     python P4Fusion.py --NoRanker             reproduces V5
     python P4Fusion.py --Members lgb lgbrank  reproduces V7
Feature history: V4 multi-space similarity + side views; V5 comparable-view SIFT evidence;
V6 LightGBM ranker blend; V7 ranker averaged over 3 seeds; V8 all-model ensemble.
Needs: pip install xgboost catboost (both on the allowed list).
Allowed methods used: gradient boosting (LightGBM, XGBoost, CatBoost, HistGradientBoosting), random
forests, logistic regression, MLP, k-NN / distance-based retrieval, bagging, simple model combination.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
RANK_SEEDS = (0, 1, 2)
MEMBERS = ("lgb", "lgbrank", "xgbrank", "hgb", "rf", "lr", "mlp", "cb", "cbrank")


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


def Learner(name):
    """classifier members (all from the allowed list)"""
    if name == "cb":
        return CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, verbose=0, random_seed=0,
                                  thread_count=2)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=50, random_state=0)
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=-999),
                             RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features=0.3,
                                                    max_samples=0.5, n_jobs=2, random_state=0))
    prep = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]
    if name == "lr":
        return make_pipeline(*prep, LogisticRegression(C=1.0, max_iter=2000))
    if name == "mlp":
        return make_pipeline(*prep, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=200,
                                                  early_stopping=True, random_state=0))
    raise ValueError(name)


def OofLearner(name, X, y, groups):
    out = np.zeros(len(X))
    for tr, va in GroupKFold(5).split(X, y, groups):
        out[va] = Learner(name).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return out


def ClassifierMember(name, V, ct, idF, qB, lab):
    """identity model + new-animal model with learner `name` (out-of-fold on validation, full fit for test)"""
    qF = qB + ["PMax", "P2nd", "PSum"]
    Vc, cc = V.copy(), ct.copy()
    y = (Vc.Identity == Vc.Truth).to_numpy(int)
    Vc["P"] = OofLearner(name, Vc[idF], y, Vc["Query"].map(lab))
    cc["P"] = Learner(name).fit(Vc[idF], y).predict_proba(cc[idF])[:, 1]
    QV, QT = QueryTable(Vc, qB), QueryTable(cc, qB)
    yNew = (Vc.groupby("QKey").Truth.first().loc[QV.index] == NEW).to_numpy(int)
    pNew = pd.Series(OofLearner(name, QV[qF], yNew, QV.index.str.split(":").str[1].map(lab)), index=QV.index)
    pNewT = pd.Series(Learner(name).fit(QV[qF], yNew).predict_proba(QT[qF])[:, 1], index=QT.index)
    return {"kind": "clf", "P": Vc.set_index(["QKey", "Identity"]).P, "pNew": pNew,
            "PT": cc.set_index(["QKey", "Identity"]).P, "pNewT": pNewT}


def FitRanker(R, feats, y, seed=0, kind="lgbrank"):
    if kind == "lgbrank":
        return lgb.LGBMRanker(**{**RANK_PARAMS, "random_state": seed}).fit(
            R[feats], y, group=R.groupby("QKey", sort=False).size().to_numpy())
    if kind == "xgbrank":
        return xgb.XGBRanker(objective="rank:map", eval_metric="map@5", tree_method="hist", n_estimators=500,
                             learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=2).fit(
            R[feats], y, qid=pd.factorize(R.QKey)[0])
    return CatBoostRanker(loss_function="YetiRank", iterations=600, learning_rate=0.05, depth=6, verbose=0,
                          random_seed=seed, thread_count=2).fit(R[feats], y, group_id=pd.factorize(R.QKey)[0])


def OofRank(R, feats, lab, seed=0, kind="lgbrank"):
    """Out-of-fold ranker scores; grouped by animal so a query's own animal never trains its scorer."""
    y = (R.Identity == R.Truth).to_numpy(int)
    out = np.zeros(len(R))
    for tr, va in GroupKFold(5).split(R, y, R["Query"].map(lab)):
        out[va] = FitRanker(R.iloc[tr], feats, y[tr], seed, kind).predict(R.iloc[va][feats])
    return out


def QueryStd(x, R):
    """per-query standardisation (puts different rankers' scores on a comparable scale)"""
    g = pd.Series(x).groupby(pd.factorize(R.QKey)[0])
    return (x - g.transform("mean").to_numpy()) / (g.transform("std").to_numpy() + 1e-9)


def RankerMember(kind, R, RT, feats, lab):
    y = (R.Identity == R.Truth).to_numpy(int)
    seeds = RANK_SEEDS if kind == "lgbrank" else (0,)
    oof = np.mean([OofRank(R, feats, lab, sd, kind) for sd in seeds], axis=0)
    test_ = np.mean([FitRanker(R, feats, y, sd, kind).predict(RT[feats]) for sd in seeds], axis=0)
    if kind != "lgbrank":
        oof, test_ = QueryStd(oof, R), QueryStd(test_, RT)
    return {"kind": "rank", "oof": oof, "test": test_}


def Softmax(R, score, shift):
    code = pd.factorize(R.QKey)[0]
    s = score + shift * (R.IsNewRow.to_numpy() == 1)
    s = s - pd.Series(s).groupby(code).transform("max").to_numpy()
    e = np.exp(s)
    return e / _GroupSum(e, code)


def V5Probs(R, pNewMap, b1, norm=True):
    """steps 1-4 as per-row probabilities (normalised per query); b1 shifts new_individual"""
    code = pd.factorize(R.QKey)[0]
    isNew = R.IsNewRow.to_numpy() == 1
    p = np.nan_to_num(R.P.to_numpy())
    pn0 = np.clip(R.QKey.map(pNewMap).to_numpy(), 1e-6, 1 - 1e-6)
    pn = 1 / (1 + np.exp(-(np.log(pn0 / (1 - pn0)) + b1)))
    v5 = np.where(isNew, pn, (1 - pn) * (p / np.maximum(_GroupSum(p * ~isNew, code), 1.0) if norm else p))
    return v5 / _GroupSum(v5, code)


def MemberProbs(m, R, test, shift, norm=True):
    """a member's per-row probabilities on the ranking rows (per query); shift moves new_individual"""
    if m["kind"] == "rank":
        return Softmax(R, m["test"] if test else m["oof"], shift)
    tmp = R.copy()
    tmp["P"] = pd.MultiIndex.from_frame(R[["QKey", "Identity"]]).map(m["PT"] if test else m["P"]).to_numpy(float)
    return V5Probs(tmp, m["pNewT"] if test else m["pNew"], shift, norm)


def BlendMembers(members, R, test, w, b, norm=True, cache=None):
    out = 0
    for n, m in members.items():
        if w[n] > 0:
            key = (n, b[n], test)
            if cache is not None and key in cache:
                p = cache[key]
            else:
                p = MemberProbs(m, R, test, b[n], norm)
                if cache is not None:
                    cache[key] = p
            out = out + w[n] * p
    return out


def TuneBlend(members, R, rate, norm, sweeps=3):
    """coordinate ascent over member weights and new_individual shifts (out-of-fold scores)"""
    names = list(members)
    w = {n: 1.0 / len(names) for n in names}
    b = {n: (0.0 if members[n]["kind"] == "clf" else -1.0) for n in names}
    cache = {}
    score = lambda w_, b_: BlendMap(R, BlendMembers(members, R, False, w_, b_, norm, cache), rate)
    best = score(w, b)
    for _ in range(sweeps):
        for n in names:
            for kind, grid in (("w", np.arange(0, 1.01, 0.1)), ("b", np.round(np.linspace(-2.5, 1.0, 15), 2))):
                for v in grid:
                    w2, b2 = dict(w), dict(b)
                    (w2 if kind == "w" else b2)[n] = v
                    if sum(w2.values()) == 0:
                        continue
                    m = score(w2, b2)
                    if m > best + 1e-6:
                        best, w, b = m, w2, b2
    return best, w, b


def _GroupSum(x, code):
    return pd.Series(x).groupby(code).transform("sum").to_numpy()


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


def Main(newBias=None, stage2=True, newRate="auto", ranker=True, members=MEMBERS):
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

    # ---- V8: ensemble of every allowed model family, blended out-of-fold ----
    feats = RankFeats(idF, qB)
    R, RT = RankRows(V, qB), RankRows(ct, qB)
    rate = newRate if newRate is not None else yNew.mean()
    M = {}
    for n in members:
        print(f"training ensemble member '{n}' out-of-fold ...", flush=True)
        if n == "lgb":  # steps 1-4 above
            M[n] = {"kind": "clf", "P": V.set_index(["QKey", "Identity"]).P, "pNew": pNew,
                    "PT": ct.set_index(["QKey", "Identity"]).P, "pNewT": pNewT}
        elif n.endswith("rank"):
            M[n] = RankerMember(n, R, RT, feats, lab)
        else:
            M[n] = ClassifierMember(n, V, ct, idF, qB, lab)
    mB, w, b = TuneBlend(M, R, rate, norm)
    tot = sum(w.values())
    print(f"ensemble CV MAP@5 re-weighted to {rate:.1%} new: {mB:.4f}\n  weights "
          f"{{{', '.join(f'{k}: {v / tot:.2f}' for k, v in w.items())}}}\n  new_individual shifts "
          f"{{{', '.join(f'{k}: {v:+.2f}' for k, v in b.items())}}}")
    WriteSubmission(TopFive(RT, BlendMembers(M, RT, True, w, b, norm)), test, train, "SubmissionFusion.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--NewBias", type=float, default=None)
    ap.add_argument("--NoStage2", action="store_true")
    ap.add_argument("--NoRanker", action="store_true", help="skip the ensemble (reproduces V5)")
    ap.add_argument("--Members", nargs="+", default=list(MEMBERS),
                    help=f"ensemble members (default: all of {' '.join(MEMBERS)}); 'lgb lgbrank' reproduces V7")
    ap.add_argument("--NewRate", default="auto",
                    help="share of new animals in test: 'auto' (estimated), a number such as 0.24, or 'none'")
    a = ap.parse_args()
    rate = None if a.NewRate == "none" else ("auto" if a.NewRate == "auto" else float(a.NewRate))
    Main(a.NewBias, not a.NoStage2, rate, not a.NoRanker, tuple(a.Members))
