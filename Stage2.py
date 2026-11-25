"""
Stage2.py -- query-to-query neighbour links (OWNER: Person 1; used by P3 for pairs, P4 for features).
The same new or known animal often appears in several test images; neighbours share evidence.
"""
import numpy as np
import pandas as pd

from Common import NEW, LoadPkl

K_NBR = 10


def QueryNeighbours(E, row, qids, sp, k=K_NBR):
    """{q: [(q2, cosine), ...]} top-k most similar OTHER queries of the same species."""
    out = {}
    for s in {sp[q] for q in qids}:
        qs = [q for q in qids if sp[q] == s]
        X = E[[row[q] for q in qs]]
        S = X @ X.T
        np.fill_diagonal(S, -9)
        idx = np.argsort(-S, axis=1)[:, :k]
        for a, q in enumerate(qs):
            out[q] = [(qs[b], float(S[a, b])) for b in idx[a]]
    return out


def AllNeighbourSets(folds, test, sp, seed=0):
    """[(prefix, neighbours)] for every validation fold and for the test set.
    The test set has about twice as many queries as a validation fold, and neighbour features
    depend on how many queries there are to search. So the test queries are split at random
    into groups the size of a validation fold, and neighbours are searched within each group."""
    sets = [(f"{f['fold']}:", QueryNeighbours(*LoadPkl(f"Emb{f['fold']}.pkl"), f["queryIds"], sp)) for f in folds]
    foldSize = np.mean([len(f["queryIds"]) for f in folds])
    ids = np.random.default_rng(seed).permutation(test.id.to_numpy())
    nGroups = max(1, int(round(len(ids) / foldSize)))
    E, row = LoadPkl("EmbTest.pkl")
    neigh = {}
    for g in np.array_split(ids, nGroups):
        neigh.update(QueryNeighbours(E, row, list(g), sp))
    sets.append(("", neigh))
    return sets


def NeighbourFeatures(c, pNew, neigh, prefix, sift):
    """Adds, per (query, identity) row:
       NbrP     max over neighbours q2 of cos(q,q2) * P(q2, identity)
       NbrPInl  max over neighbours with >=15 SIFT inliers to q of P(q2, identity)
       and per query: NbrCos (best neighbour cosine), NbrInl (best neighbour inliers),
       NbrNew (pNew of the most strongly linked neighbour)."""
    P = {(q, i): p for q, i, p in zip(c["Query"], c["Identity"], c["P"])}
    links, qCos, qInl, qNew = {}, {}, {}, {}
    for q in c["Query"].unique():
        lk = [(q2, cs, max(sift.get((q, q2), (0, 0))[1], sift.get((q2, q), (0, 0))[1]))
              for q2, cs in neigh.get(q, [])]
        links[q] = lk
        qCos[q] = max((l[1] for l in lk), default=np.nan)
        qInl[q] = max((l[2] for l in lk), default=0)
        st = max(lk, key=lambda l: (l[2], l[1]), default=None)
        qNew[q] = pNew.get(prefix + st[0], np.nan) if st else np.nan
    c = c.copy()
    c["NbrP"] = [max((cs * P.get((q2, i), 0.0) for q2, cs, _ in links[q]), default=0.0)
                 for q, i in zip(c["Query"], c["Identity"])]
    c["NbrPInl"] = [max((P.get((q2, i), 0.0) for q2, _, inl in links[q] if inl >= 15), default=0.0)
                    for q, i in zip(c["Query"], c["Identity"])]
    c["NbrCos"] = c["Query"].map(qCos)
    c["NbrInl"] = c["Query"].map(qInl)
    c["NbrNew"] = c["Query"].map(qNew)
    return c


def QueryPairs(sets):
    """SIFT pairs needed for the neighbour links."""
    return {(q, q2) for _, nb in sets for q, lst in nb.items() for q2, _ in lst}
