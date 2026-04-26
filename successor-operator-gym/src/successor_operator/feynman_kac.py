"""Exact Feynman-Kac / successor-mass solver.

Solves the linear system (I - M_w) h_N = c_w for the desirability function h_w.
Vectorized implementation using numpy broadcasting for performance.
"""

import numpy as np
from scipy.linalg import solve


def solve_successor_mass(
    mdp,
    pi0: np.ndarray,
    step_feat: np.ndarray,
    term_feat: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Compute the exact successor mass h_w(s) for all states.

    Uses the vectorized linear system (I - M_w) h_N = c_w.

    Args:
        mdp: TabularMDP instance.
        pi0: Passive policy, shape [nS, nA].
        step_feat: Step features, shape [nS, nA, nS, d].
        term_feat: Terminal features, shape [nS, d].
        w: Task weight vector, shape [d].

    Returns:
        h: Successor mass, shape [nS].
    """
    nS = mdp.nS
    nonterm = mdp.nonterminal_indices()
    term = mdp.terminal_indices()

    h = np.zeros(nS, dtype=np.float64)
    if len(term) > 0:
        h[term] = np.exp(term_feat[term] @ w)

    if len(nonterm) == 0:
        return h

    # Vectorized construction of M and c
    # exp_weights[s, a, s'] = pi0[s,a] * P[s,a,s'] * exp(step_feat[s,a,s'] @ w)
    dot_products = np.einsum("ijkl,l->ijk", step_feat, w)
    exp_weights = pi0[:, :, None] * mdp.P * np.exp(dot_products)

    # M_w[s, s'] = sum_a exp_weights[s, a, s'] for s, s' nonterminal
    full_M = exp_weights.sum(axis=1)  # [nS, nS]
    M = full_M[np.ix_(nonterm, nonterm)]

    # c_w[s] = sum over terminal s': full_M[s, s'] * h[s']
    if len(term) > 0:
        c = full_M[np.ix_(nonterm, term)] @ h[term]
    else:
        c = np.zeros(len(nonterm), dtype=np.float64)

    A = np.eye(len(nonterm)) - M
    h_non = solve(A, c, assume_a="gen")
    h[nonterm] = h_non
    return h


def solve_successor_mass_sparse(
    mdp,
    pi0: np.ndarray,
    step_feat: np.ndarray,
    term_feat: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Sparse version for larger state spaces (e.g., Taxi with 500 states).

    Uses scipy.sparse for the linear system.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    nS = mdp.nS
    nonterm = mdp.nonterminal_indices()
    term = mdp.terminal_indices()

    h = np.zeros(nS, dtype=np.float64)
    if len(term) > 0:
        h[term] = np.exp(term_feat[term] @ w)

    if len(nonterm) == 0:
        return h

    idx_map = np.full(nS, -1, dtype=int)
    idx_map[nonterm] = np.arange(len(nonterm))

    rows, cols, vals = [], [], []
    c = np.zeros(len(nonterm), dtype=np.float64)

    for s in nonterm:
        i = idx_map[s]
        for a in range(mdp.nA):
            pa = pi0[s, a]
            if pa == 0.0:
                continue
            for s2 in range(nS):
                p = mdp.P[s, a, s2]
                if p == 0.0:
                    continue
                weight = pa * p * np.exp(step_feat[s, a, s2] @ w)
                if mdp.terminal[s2]:
                    c[i] += weight * h[s2]
                else:
                    rows.append(i)
                    cols.append(idx_map[s2])
                    vals.append(weight)

    n = len(nonterm)
    M_sparse = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_sparse = csr_matrix(np.eye(n)) - M_sparse
    h_non = spsolve(A_sparse, c)
    h[nonterm] = h_non
    return h
