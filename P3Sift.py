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
