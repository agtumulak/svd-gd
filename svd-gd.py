#!/usr/bin/env python

import numpy as np
import pandas as pd
import tikzplotlib
from functools import partial
from scipy.optimize import minimize, NonlinearConstraint
np.random.seed(42)

from matplotlib import pyplot as plt


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
    r_full = S.size
    U = U.unstack()
    S = pd.DataFrame(np.diag(S.values.flatten()), index=U.columns, columns=U.columns)
    Vt = V.unstack().T
    A_full = U @ S @ Vt
    # get truncated rank-r matrix
    r = 3
    U_truncated = U.iloc[:,:r]
    S_truncated = S.iloc[:r,:r]
    Vt_truncated = Vt.iloc[:r,:]
    A_truncated = U_truncated @ S_truncated @ Vt_truncated
    nonmonoticity = partial(
            row_nonmonoticity,
            A_full.columns.unique('E').size,
            U_truncated.values,
            Vt_truncated.values)
    print(
            f"truncation l2-error: {np.linalg.norm(A_full - A_truncated) / np.linalg.norm(A_full)}\n"
            f"truncation l-inf-error: {np.linalg.norm(A_full - A_truncated, ord=np.inf) / np.linalg.norm(A_full, ord=np.inf)}\n"
            f"nonmonoticity error: {nonmonoticity(np.diag(S_truncated))}\n"
            f"singular values: {np.diag(S_truncated)}")
    # use unconstrained truncated SVD as initial guess
    res = minimize(
            partial(
                lambda A_full, U, Vt, s_vals:
                np.linalg.norm(U @ np.diag(s_vals) @ Vt - A_full),
                A_full, U_truncated.values, Vt_truncated.values),
            np.diag(S_truncated),
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
    S_optimized = pd.DataFrame(np.diag(res.x), index=U_truncated.columns, columns=Vt_truncated.index)
    A_optimized = U_truncated @ S_optimized @ Vt_truncated
    print(
            f"monoticity-optimized error: {np.linalg.norm(A_full - A_optimized) / np.linalg.norm(A_full)}\n"
            f"truncation l-inf-error: {np.linalg.norm(A_full - A_optimized, ord=np.inf) / np.linalg.norm(A_full, ord=np.inf)}\n"
            f"monoticity-optimized monoticity error: {nonmonoticity(res.x)}\n"
            f"singular values: {np.diag(S_optimized)}")

    # figure out nonmonotonic column
    plotting = True
    if plotting:
        cdf_form_full = A_full.stack('CDF').unstack('T')
        cdf_form_truncated = A_truncated.stack('CDF').unstack('T')
        cdf_form_optimized = A_optimized.stack('CDF').unstack('T')
        most_nonmonotonic_idx = (cdf_form_truncated.diff() < 0).sum().argmax()
        most_nonmonotonic_full = cdf_form_full.iloc[:, most_nonmonotonic_idx]
        most_nonmonotonic_truncated = cdf_form_truncated.iloc[:, most_nonmonotonic_idx]
        most_nonmonotonic_optimized = cdf_form_optimized.iloc[:, most_nonmonotonic_idx]

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(
                most_nonmonotonic_full.values,
                most_nonmonotonic_truncated.index,
                label=f'$r={r_full}$',
                color='k',
                linestyle='solid')
        axs[0].plot(
                most_nonmonotonic_truncated.values,
                most_nonmonotonic_truncated.index,
                label=f'$r={r}$',
                color='k',
                linestyle='dashed')
        axs[0].plot(
                most_nonmonotonic_optimized.values,
                most_nonmonotonic_optimized.index,
                label=f'$r={r}$, opt.',
                color='k',
                linestyle='dotted')
        axs[0].set_xlim(-1, 20)
        axs[0].set_ylim(0, 1.1)
        axs[0].set_xlabel("$\\beta$")
        axs[0].set_ylabel("$G \\left( \\beta \\mid E, T \\right)$")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(
                most_nonmonotonic_full.values,
                most_nonmonotonic_truncated.index,
                label=f'$r={r_full}$',
                color='k',
                linestyle='solid')
        axs[1].plot(
                most_nonmonotonic_truncated.values,
                most_nonmonotonic_truncated.index,
                label=f'$r={r}$',
                color='k',
                linestyle='dashed')
        axs[1].plot(
                most_nonmonotonic_optimized.values,
                most_nonmonotonic_optimized.index,
                label=f'$r={r}$, opt.',
                color='k',
                linestyle='dotted')
        axs[1].set_xlim(-0.025, 0.025)
        axs[1].set_ylim(0, 0.6)
        axs[1].set_xlabel("$\\beta$")
        axs[1].legend()
        axs[1].grid()

        print(f"E: {most_nonmonotonic_truncated.name[0] * 1e6} eV")
        print(f"T: {most_nonmonotonic_truncated.name[1]} K")
        tikzplotlib.clean_figure()
        tikzplotlib.save("/Users/atumulak/Developer/phd-dissertation/figures/beta-most-nonmonotonic-cdf-fullorder.tex")
