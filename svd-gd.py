#!/usr/bin/env python

import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize, NonlinearConstraint
np.random.seed(42)


from ipdb import set_trace as st


def row_nonmonoticity(n_Es, U, Vt, x):
    """
    For use with scipy.optimize.NonlinearConstraint
    """
    # convert A_truncated to a numpy.ndarray where the innermost index
    # corresponds to CDF values then compute first differences
    diffs = np.diff(np.array(np.split(U @ np.diag(x) @ Vt, n_Es, axis=1)))
    return np.linalg.norm(diffs[diffs < 0])


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
    # get truncated rank-r matrix
    r = 3
    U = U.iloc[:,:r]
    S = S.iloc[:r,:r]
    Vt = Vt.iloc[:r,:]
    A_truncated = U @ S @ Vt
    nonmonoticity = partial(row_nonmonoticity, A_full.columns.unique('E').size, U.values, Vt.values)
    print(
            f"truncation l2-error: {np.linalg.norm(A_full - A_truncated) / np.linalg.norm(A_full)}\n"
            f"truncation l-inf-error: {np.linalg.norm(A_full - A_truncated, ord=np.inf) / np.linalg.norm(A_full, ord=np.inf)}\n"
            f"nonmonoticity error: {nonmonoticity(np.diag(S))}\n"
            f"singular values: {np.diag(S)}")
    # use unconstrained truncated SVD as initial guess
    res = minimize(
            partial(
                lambda A_full, U, Vt, s_vals:
                np.linalg.norm(U @ np.diag(s_vals) @ Vt - A_full),
                A_full, U.values, Vt.values),
            np.diag(S),
            method='trust-constr',
            constraints=NonlinearConstraint(
                nonmonoticity,
                -np.inf,
                0,
                jac='3-point'),
            options={
                'verbose': 2,
                'disp': True})
    # reconstruct
    S = pd.DataFrame(np.diag(res.x), index=U.columns, columns=Vt.index)
    A_optimized = U @ S @ Vt
    print(
            f"monoticity-optimized error: {np.linalg.norm(A_full - A_optimized) / np.linalg.norm(A_full)}\n"
            f"truncation l-inf-error: {np.linalg.norm(A_full - A_optimized, ord=np.inf) / np.linalg.norm(A_full, ord=np.inf)}\n"
            f"monoticity-optimized monoticity error: {nonmonoticity(res.x)}\n"
            f"singular values: {np.diag(S)}")
