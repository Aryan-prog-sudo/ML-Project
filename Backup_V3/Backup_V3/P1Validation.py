"""
PERSON 1 -- validation folds, metric, submissions, query-to-query propagation.
    python P1Validation.py Folds        # FIRST -> Work/Folds.pkl
    python P1Validation.py Propagate    # after P2Global.py
TODO:
  [ ] submit all-new_individual once: public score = fraction of new on public LB;
      compare with the fold fractions printed by `Folds`
  [ ] keep a LB log (file, CV, public) to spot CV/LB gaps
  [ ] re-run propagation on P4's probabilities instead of cosines
"""
import sys

import numpy as np
import pandas as pd

from Common import (NEW, WORK, DATA, LoadMeta, IdMaps, MakeFolds, SavePkl, LoadFolds,
                    LoadPkl, Map5, CheckSubmission)


ENC_TAU = 0.8     # photos of one animal with whitened cosine > ENC_TAU form one encounter
UNK_FRAC = 0.2    # share of identities hidden per fold (controls the new_individual rate)


def Encounters(train, tau=ENC_TAU):
    """Graph clustering (connected components) of each identity's photos: photos linked when
    their whitened MiewID cosine > tau, i.e. near-identical shots from the same burst."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from Common import LoadMiewid
    from P2Global import Whiten
    sp, _, _ = IdMaps(train, train.iloc[:0])
    F, row = LoadMiewid()
    W = Whiten(F, row, {**sp, **dict(zip(LoadMeta()[1].id, LoadMeta()[1].species))})
    enc = {}
    for ident, g in train.groupby("identity"):
        ids = g.id.tolist()
        X = W[[row[i] for i in ids]]
        _, comp = connected_components(csr_matrix(X @ X.T > tau), directed=False)
        enc.update({i: f"{ident}#{c}" for i, c in zip(ids, comp)})
    return enc


def CmdFolds():
    train, test = LoadMeta()
    sp, _, _ = IdMaps(train, test)
    enc = Encounters(train)
    print(f"{len(set(enc.values()))} encounters among {len(train)} train photos")
    folds = MakeFolds(train, nFolds=4, unkFrac=UNK_FRAC, enc=enc)
    SavePkl(folds, "Folds.pkl")
    for f in folds:
        t = pd.Series(f["truth"])
        print(f"fold {f['fold']}: {len(t)} queries | new {np.mean(t == NEW):.1%} | "
              f"gallery {len(f['galleryIds'])} | hyena queries {sum(sp[q] == 'hyena' for q in t.index)}")
        # sanity: predicting only new_individual scores exactly the new fraction
        assert abs(Map5(f["truth"], {q: [NEW] for q in t.index}) - np.mean(t == NEW)) < 1e-9
    CheckSubmission(pd.read_csv(DATA / "sample_submission.csv"), train, test)
    print("sanity checks passed -> Work/Folds.pkl")


def ScoresFromCands(c, newScore):
    """P2 candidate table -> {query: {label: score}} with a constant score for NEW."""
    out = {}
    for q, lab, s in zip(c["Query"], c["Identity"], c["CosMax"]):
        out.setdefault(q, {NEW: newScore})[lab] = s
    return out


def Propagate(scores, neigh, thr):
    """q inherits the identity scores of a very similar other query q2 (sim >= thr),
    discounted by that similarity. NEW is never propagated."""
    out = {}
    for q, sc in scores.items():
        s2 = dict(sc)
        for q2, sim in neigh.get(q, []):
            if sim < thr or q2 not in scores:
                continue
            for lab, v in scores[q2].items():
                if lab != NEW:
                    s2[lab] = max(s2.get(lab, -np.inf), sim * v)
        out[q] = s2
    return out


def Rank(scores):
    return {q: [l for l, _ in sorted(sc.items(), key=lambda t: -t[1])[:5]] for q, sc in scores.items()}


def QueryNeighbours(E, row, qids, sp, k=10):
    """Top-k most similar OTHER queries of the same species."""
    out = {}
    for s in {sp[q] for q in qids}:
        qs = [q for q in qids if sp[q] == s]
        X = E[[row[q] for q in qs]]
        S = X @ X.T
        np.fill_diagonal(S, -1)
        idx = np.argsort(-S, axis=1)[:, :k]
        for a, q in enumerate(qs):
            out[q] = [(qs[b], float(S[a, b])) for b in idx[a]]
    return out


def CmdPropagate():
    train, test = LoadMeta()
    sp, _, _ = IdMaps(train, test)
    folds = LoadFolds()
    cands = {f["fold"]: pd.read_pickle(WORK / f"CandsFold{f['fold']}.pkl") for f in folds}
    neigh = {f["fold"]: QueryNeighbours(*LoadPkl(f"Emb{f['fold']}.pkl"), f["queryIds"], sp) for f in folds}
    allCos = np.concatenate([c.CosMax.to_numpy() for c in cands.values()])
    grid = np.quantile(allCos, np.linspace(0.05, 0.99, 25))  # candidate scores for NEW
    for thr in [None, 0.95, 0.9, 0.85, 0.8, 0.75]:
        best = (0, None)
        for ns in grid:
            m = []
            for f in folds:
                sc = ScoresFromCands(cands[f["fold"]], ns)
                if thr is not None:
                    sc = Propagate(sc, neigh[f["fold"]], thr)
                m.append(Map5(f["truth"], Rank(sc)))
            best = max(best, (np.mean(m), ns))
        print(f"thr={thr}: CV MAP@5 {best[0]:.4f} (new score {best[1]:.2f})")


if __name__ == "__main__":
    {"Folds": CmdFolds, "Propagate": CmdPropagate}[sys.argv[1] if len(sys.argv) > 1 else "Folds"]()
