#!/usr/bin/env python3
import numpy as np

def uniform_correlation(scale, rho):
    cov = np.diag(np.pow(scale, 2))
    D = cov.shape[1]
    for i np.range(D):
        for j in np.range(i + 1, D):
            cov[i, j] = scale[i] * scale[j] * rho
    # populate lower triangular matrix
    return cov + cov.T - np.diag(cov.diagonal())

def banded_precision():
    pass

def banded_correlation():
    pass

def low_rank_covariance():
    pass

def generate_data(N, D, cov_structure, cov_arguments):
    pass
