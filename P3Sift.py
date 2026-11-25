"""
PERSON 3 -- SIFT matching + RANSAC.
Cache: Work/SiftPairs.pkl {(queryId, galleryId): (nGood, nInliers)},  Work/Nkp.pkl {id: n keypoints}
    python P3Sift.py              # pairs from P2's Work/Cands*.pkl + query-query neighbour pairs (run P2 first)
    python P3Sift.py --Quick      # start without P2: pairs from raw cosine
    python P3Sift.py --TestTest   # test<->test pairs (for P1's propagation)
    python P3Sift.py --Diagnose   # inlier counts: same vs different identity
Resumable: already-computed pairs are skipped.
TODO:
  [ ] tune RATIO (0.75-0.85) and RANSAC_PX (8-20) with --Diagnose
  [ ] try cv2.estimateAffine2D / findFundamentalMat (bodies bend)
  [ ] extra features for P4: inlier spatial spread, mean match distance
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")  # we parallelise with threads ourselves

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from Common import WORK, LoadMeta, IdMaps, LoadSift, LoadMiewid, LoadPkl, SavePkl

cv2.setNumThreads(1)
RATIO, RANSAC_PX = 0.8, 15.0


def RootSift(d):
    d = d.astype(np.float32)
    d /= d.sum(1, keepdims=True) + 1e-7
    return np.sqrt(d)  # rows now have unit L2 norm


def MatchPair(qxy, qd, gxy, gd):
    """qd, gd are RootSIFT. Returns (nGood ratio-test matches, nInliers after RANSAC)."""
    if len(qd) < 2 or len(gd) < 2:
        return 0, 0
    S = qd @ gd.T                       # cosine; dist^2 = 2 - 2*cos
    r = np.arange(len(S))
    j1 = S.argmax(1)
    s1 = S[r, j1].copy()
    S[r, j1] = -np.inf
    s2 = S.max(1)
    d1, d2 = np.sqrt(np.maximum(2 - 2 * s1, 0)), np.sqrt(np.maximum(2 - 2 * s2, 0))
    qi = np.flatnonzero(d1 < RATIO * d2)    # Lowe ratio test
    gj = j1[qi]
    if len(qi):                             # one-to-one: best match per gallery keypoint
        o = np.argsort(d1[qi])
        qi, gj = qi[o], gj[o]
        _, u = np.unique(gj, return_index=True)
        qi, gj = qi[u], gj[u]
    if len(qi) < 4:
        return len(qi), 0
    _, inl = cv2.estimateAffinePartial2D(qxy[qi], gxy[gj], method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_PX, maxIters=2000, confidence=0.995)
    return len(qi), int(inl.sum()) if inl is not None else 0

def PairsFromCandidates():
    """Query x candidate-image pairs (from P2) plus query x neighbouring-query pairs (for P4 stage 2)."""
    files = sorted(WORK.glob("Cands*.pkl"))
    if not files:
        raise SystemExit("no Work/Cands*.pkl -- run P2Global.py first or use --Quick")
    pairs = set()
    for f in files:
        c = pd.read_pickle(f)
        for q, imgs in zip(c["Query"], c["BestImgs"]):
            pairs.update((q, g) for g in imgs)
    import Stage2
    train, test = LoadMeta()
    sp, _, _ = IdMaps(train, test)
    pairs |= Stage2.QueryPairs(Stage2.AllNeighbourSets(LoadPkl("Folds.pkl"), test, sp))
    return pairs


def PairsQuick(k=30, nTrainQ=1500, testTest=False):
    """No dependency on P2: top-k raw-cosine neighbours."""
    train, test = LoadMeta()
    sp, _, _ = IdMaps(train, test)
    F, row = LoadMiewid()
    E = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    tq = list(np.random.default_rng(0).choice(train.id.to_numpy(), min(nTrainQ, len(train)), replace=False))
    pairs = set()
    for s in ("hyena", "leopard"):
        if testTest:
            queries = gal = [q for q in test.id if sp[q] == s]
        else:
            gal = train.id[train.species == s].tolist()
            queries = [q for q in tq if sp[q] == s] + [q for q in test.id if sp[q] == s]
        G = E[[row[g] for g in gal]]
        for b in range(0, len(queries), 512):
            qb = queries[b:b + 512]
            S = E[[row[q] for q in qb]] @ G.T
            for r, q in enumerate(qb):
                pairs.update((q, gal[j]) for j in np.argsort(-S[r])[:k + 1] if gal[j] != q)
    return pairs


def Run(pairs, workers):
    cache = LoadPkl("SiftPairs.pkl", default={})
    todo = [p for p in pairs if p not in cache]
    print(f"{len(cache)} cached, {len(todo)} to compute", flush=True)
    if not todo:
        return
    need = sorted({i for p in todo for i in p})
    t0 = time.time()
    store = {i: LoadSift(i) for i in need}      # main thread
    nkp = LoadPkl("Nkp.pkl", default={})
    nkp.update({i: len(v[1]) for i, v in store.items()})
    SavePkl(nkp, "Nkp.pkl")
    print(f"loaded SIFT for {len(need)} images in {time.time() - t0:.0f}s", flush=True)
    byQ = {}
    for q, g in todo:
        byQ.setdefault(q, []).append(g)

    def Task(item):
        q, gs = item
        qxy, qd = store[q][0], RootSift(store[q][1])
        return [((q, g), MatchPair(qxy, qd, store[g][0], RootSift(store[g][1]))) for g in gs]

    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(Task, byQ.items(), chunksize=8):
            cache.update(res)
            done += len(res)
            if done % 20000 < len(res):
                print(f"{done}/{len(todo)} pairs, {done / (time.time() - t0):.0f} pairs/s", flush=True)
                SavePkl(cache, "SiftPairs.pkl")
    SavePkl(cache, "SiftPairs.pkl")
    print(f"done: {len(cache)} pairs cached in Work/SiftPairs.pkl", flush=True)


def Diagnose():
    train, test = LoadMeta()
    _, _, lab = IdMaps(train, test)
    cache = LoadPkl("SiftPairs.pkl")
    d = pd.DataFrame([(v[1], lab[q] == lab[g]) for (q, g), v in cache.items() if q in lab and g in lab],
                     columns=["Inliers", "Same"])
    print(d.groupby("Same").Inliers.describe(percentiles=[.5, .9, .99]))
    for t in (5, 8, 10, 15, 20, 30):
        s = d[d.Inliers >= t]
        print(f">= {t:2d} inliers: {len(s):6d} pairs, precision {s.Same.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Quick", action="store_true")
    ap.add_argument("--TestTest", action="store_true")
    ap.add_argument("--Diagnose", action="store_true")
    ap.add_argument("--Workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if a.Diagnose:
        Diagnose()
    elif a.TestTest:
        Run(PairsQuick(k=10, testTest=True), a.Workers)
    else:
        Run(PairsQuick() if a.Quick else PairsFromCandidates(), a.Workers)