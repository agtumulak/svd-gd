#!/usr/bin/env python

import numpy as np
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
np.random.seed(45)


def encode(U, S, Vt):
    return np.concatenate((
        U.values.reshape(-1),
        np.diag(S.values),
        Vt.values.reshape(-1)))


def decode(x, m, n, r):
    """
    Decodes flattened vector into singular value decomposition coefficients

    Parameters:
    x : numpy.ndarray
        Singular value decomposition encoded as a 1D vector.
    m : int
        Number of rows in matrix
    n : int
        Number of columns in matrix
    r : int
        Rank of matrix
    """
    U = x[:m * r].reshape(m, r)
    S = np.diag(x[m * r: m * r + r])
    Vt = x[m * r + r:].reshape(r, n)
    return U, S, Vt


def nonunitariness(B):
    B = B @ B.T
    return np.linalg.norm(B - np.identity(B.shape[0]))


def cdf_rows(A_truncated, A_full):
    """
    Convert A_truncated to a numpy.ndarray where the innermost index
    corresponds to CDF values
    """
    n_Es = A_full.columns.unique('E').size
    return np.array(np.split(A_truncated, n_Es, axis=1))


def row_nonmonoticity(A_truncated, A_full):
    """
    Parameters
    ----------

    A_truncated : numpy.ndarray
        The reconstructed matrix whose monmonocity is to be measured
    A_full : pandas.DataFrame
        The original untruncated matrix whose metadata is to be used for
        interpreting A_truncated
    """
    # # The columns are a flattened index of incident energy and CDF values.
    # # Since we are measuring nonmonoticity in CDF (and not energy), we need
    # # to compute the number of unique incidient energies to split the data.
    # is_monotonic = np.diff(cdf_rows(A_truncated, A_full)) > 0
    # # instead of measuring differences in beta, we measure differences in CDF
    # # since differences in CDF correspond to areas under the PDF
    # pdf_areas = np.diff(A_full.columns.unique('CDF'))
    # negative_pdf_mass = np.where(~is_monotonic, pdf_areas, 0)
    # # normalize by number of incident energies and CDFs so that returned value
    # # is always in [0, 1]
    # n_Es = A_full.columns.unique('E').size
    # n_Ts = A_full.index.unique().size
    # return np.sum(negative_pdf_mass) / (n_Es * n_Ts)
    diffs = np.diff(cdf_rows(A_truncated, A_full))
    return np.linalg.norm(diffs[diffs < 0])


def row_nonmonoticity_constraint(A_full, r, x):
    """
    For use with scipy.optimize.NonlinearConstraint
    """
    m, n = A_full.shape
    U, S, Vt = decode(x, m, n, r)
    A = U @ S @ Vt
    return [row_nonmonoticity(A, A_full)]


def cost(x, A_full, r):
    """
    Objective function for scipy.optimize.minimze

    Parameters
    ----------
    x : numpy.ndarray
        Singular value decomposition encoded as a 1D vector
    A_full : numpy.ndarray
        The original untruncated matrix
    r : int
        Rank of matrix
    """
    m, n = A_full.shape
    U, S, Vt = decode(x, m, n, r)
    return np.linalg.norm(U @ S @ Vt - A_full)


if __name__ == '__main__':
    # load TSL data
    U = pd.read_hdf('~/Developer/minimc/data/tsl/endfb8-fullorder/beta_endfb8_T_coeffs.hdf5')
    S = pd.read_hdf('~/Developer/minimc/data/tsl/endfb8-fullorder/beta_endfb8_S_coeffs.hdf5')
    V = pd.read_hdf('~/Developer/minimc/data/tsl/endfb8-fullorder/beta_endfb8_E_CDF_coeffs.hdf5')
    # reconstruct full-rank matrix
    U = U.unstack()
    S = pd.DataFrame(np.diag(S.values.flatten()), index=U.columns, columns=U.columns)
    Vt = V.unstack().T
    A_full = U @ S @ Vt
    # keep subset of energies, CDFs
    Es_all = A_full.columns.unique('E')
    Es_subset = Es_all[np.round(np.linspace(0, Es_all.size-1, 20)).astype(int)]
    CDFs_all = A_full.columns.unique('CDF')
    CDFs_subset = CDFs_all[np.round(np.linspace(0, CDFs_all.size-1, 15)).astype(int)]
    column_index = pd.MultiIndex.from_product((Es_subset, CDFs_subset), names=('E', 'CDF'))
    row_index = A_full.index
    order_index = pd.Index(range(min(row_index.size, column_index.size)), name='order')
    A_full = A_full.loc[row_index, column_index]
    print(f"shape: {A_full.shape}")
    # perform SVD on subsetted data
    U, S, Vt = np.linalg.svd(A_full, full_matrices=False)
    U = pd.DataFrame(U, index=row_index, columns=order_index)
    S = pd.DataFrame(np.diag(S), index=order_index, columns=order_index)
    Vt = pd.DataFrame(Vt, index=order_index, columns=column_index)
    # get truncated rank-r matrix
    r = 3
    U = U.iloc[:,:r]
    S = S.iloc[:r,:r]
    Vt = Vt.iloc[:r,:]
    A_truncated = U @ S @ Vt
    # compute nonmonoticity penalty so that truncation error is on the same order
    # as nonmonoticity error
    print(
            f"truncation error: {np.linalg.norm(A_full - A_truncated) / np.linalg.norm(A_full)}\n"
            f"nonmonoticity error: {row_nonmonoticity(A_truncated, A_full)}")
    # use unconstrained truncated SVD as initial guess
    res = minimize(
            cost,
            encode(U, S, Vt), # use unconstrained truncated SVD as initial guess
            args=(A_full, r),
            method='trust-constr',
            constraints=NonlinearConstraint(
                partial(row_nonmonoticity_constraint, A_full, r),
                -np.inf,
                0,
                jac='3-point'),
            options={
                'verbose': 2,
                'disp': True})
    U, S, Vt = decode(res.x, U.shape[0], Vt.shape[1], r)
    # singular values are conventionally in decreasing order
    # inverse of P is P.T (unitary)
    P = np.zeros(S.shape)
    for index, value in enumerate(np.argsort(-np.abs(np.diag(S)))):
        P[value, index] = 1
    # singular values are conventionally positive
    # inverse of Q is Q (signature matrices are involutory)
    Q = np.sign(S)
    # apply adjustments
    U = U @ P @ Q
    S = Q @ P.T @ S @ P
    Vt = P.T @ Vt
    # reconstruct
    A_optimized = U @ S @ Vt
    print(f"optimized truncated error: {np.linalg.norm(A_full - A_optimized) / np.linalg.norm(A_full)}")
    print(f"optimized truncated nonmonoticity: {row_nonmonoticity(A_optimized, A_full)}")
