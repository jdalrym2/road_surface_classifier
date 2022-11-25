#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python implementation of geometric median as described in
    Yehuda Vardi and Cun-Hui Zhang's paper 
    "The multivariate L1-median and associated data depth"
"""
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Python implementation of geometric median as described in
    Yehuda Vardi and Cun-Hui Zhang's paper 
    "The multivariate L1-median and associated data depth"

    Taken from: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    Released under zlib license.

    Args:
        X (np.ndarray): Input N-dimensional data
        eps (float, optional): Convergence tolerance. Defaults to 1e-5.

    Returns:
        np.ndarray: Geometric median value
    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1