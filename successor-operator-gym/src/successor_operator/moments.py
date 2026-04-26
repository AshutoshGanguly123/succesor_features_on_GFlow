"""First and second moment computation for local free-energy geometry.

Computes gradient g_w(s) = ∇_w A_w(s) and Hessian H_w(s) = ∇²_w A_w(s)
via the exact U/V recursions, then derives grad and hess from h.
"""

import numpy as np
from scipy.linalg import solve


def solve_moments(mdp, pi0, step_feat, term_feat, w, h):
    """Compute exact gradient and Hessian of log h_w for all states.

    Uses vectorized linear systems where possible.

    Args:
        mdp: TabularMDP instance.
        pi0: Passive policy, shape [nS, nA].
        step_feat: Step features, shape [nS, nA, nS, d].
        term_feat: Terminal features, shape [nS, d].
        w: Task weight vector, shape [d].
        h: Successor mass from solve_successor_mass, shape [nS].

    Returns:
        grad: shape [nS, d], gradient of log h_w.
        hess: shape [nS, d, d], Hessian of log h_w.
    """
    nS = mdp.nS
    d = step_feat.shape[-1]
    nonterm = mdp.nonterminal_indices()
    term = mdp.terminal_indices()

    # Pre-compute the shared linear operator M
    dot_products = np.einsum("ijkl,l->ijk", step_feat, w)
    exp_weights = pi0[:, :, None] * mdp.P * np.exp(dot_products)
    full_M = exp_weights.sum(axis=1)
    M = full_M[np.ix_(nonterm, nonterm)]
    A_mat = np.eye(len(nonterm)) - M

    # Initialize U and V
    U = np.zeros((nS, d), dtype=np.float64)
    V = np.zeros((nS, d, d), dtype=np.float64)

    if len(term) > 0:
        term_exp = np.exp(term_feat[term] @ w)
        U[term] = term_feat[term] * term_exp[:, None]
        for k, s in enumerate(term):
            beta = term_feat[s]
            V[s] = np.outer(beta, beta) * np.exp(beta @ w)

    # Solve U for all dimensions at once using a batched RHS
    # b_dim[i, dim] = sum_a sum_s' exp_weight[s,a,s'] * (psi_dim[s,a,s']*h[s'] + U[s',dim])
    rhs_U = np.zeros((len(nonterm), d), dtype=np.float64)
    for dim in range(d):
        for idx_i, s in enumerate(nonterm):
            val = 0.0
            for a in range(mdp.nA):
                pa = pi0[s, a]
                if pa == 0.0:
                    continue
                for s2 in range(nS):
                    p = mdp.P[s, a, s2]
                    if p == 0.0:
                        continue
                    ew = pa * p * np.exp(step_feat[s, a, s2] @ w)
                    val += ew * (step_feat[s, a, s2, dim] * h[s2] + U[s2, dim])
            rhs_U[idx_i, dim] = val

    U_non = solve(A_mat, rhs_U, assume_a="gen")
    U[nonterm] = U_non

    # Solve V dimension by dimension
    for i_dim in range(d):
        for j_dim in range(i_dim, d):
            rhs = np.zeros(len(nonterm), dtype=np.float64)
            for idx_i, s in enumerate(nonterm):
                val = 0.0
                for a in range(mdp.nA):
                    pa = pi0[s, a]
                    if pa == 0.0:
                        continue
                    for s2 in range(nS):
                        p = mdp.P[s, a, s2]
                        if p == 0.0:
                            continue
                        ew = pa * p * np.exp(step_feat[s, a, s2] @ w)
                        psi_i = step_feat[s, a, s2, i_dim]
                        psi_j = step_feat[s, a, s2, j_dim]
                        cont = (
                            V[s2, i_dim, j_dim]
                            + psi_i * U[s2, j_dim]
                            + psi_j * U[s2, i_dim]
                            + psi_i * psi_j * h[s2]
                        )
                        val += ew * cont
                rhs[idx_i] = val

            V_col = solve(A_mat, rhs, assume_a="gen")
            V[nonterm, i_dim, j_dim] = V_col
            if i_dim != j_dim:
                V[nonterm, j_dim, i_dim] = V_col

    h_safe = np.maximum(h, 1e-300)
    grad = U / h_safe[:, None]
    hess = V / h_safe[:, None, None] - grad[:, :, None] * grad[:, None, :]
    return grad, hess


def solve_moments_vectorized(mdp, pi0, step_feat, term_feat, w, h):
    """Fully vectorized moment computation. Faster for small-medium state spaces.

    Uses einsum + batched linear solves instead of Python loops over states.
    """
    nS = mdp.nS
    d = step_feat.shape[-1]
    nonterm = mdp.nonterminal_indices()
    term = mdp.terminal_indices()

    dot_products = np.einsum("ijkl,l->ijk", step_feat, w)
    exp_weights = pi0[:, :, None] * mdp.P * np.exp(dot_products)

    # exp_weights: [nS, nA, nS]
    # step_feat: [nS, nA, nS, d]
    # h: [nS]

    full_M = exp_weights.sum(axis=1)
    M = full_M[np.ix_(nonterm, nonterm)]
    A_mat = np.eye(len(nonterm)) - M

    U = np.zeros((nS, d), dtype=np.float64)
    V = np.zeros((nS, d, d), dtype=np.float64)

    if len(term) > 0:
        term_exp = np.exp(term_feat[term] @ w)
        U[term] = term_feat[term] * term_exp[:, None]
        for s in term:
            beta = term_feat[s]
            V[s] = np.outer(beta, beta) * np.exp(beta @ w)

    # U RHS: sum_a sum_s' ew[s,a,s'] * (psi_dim[s,a,s'] * h[s'] + U[s', dim])
    # First part: sum_a sum_s' ew[s,a,s'] * psi[s,a,s',dim] * h[s']
    ew_h = exp_weights * h[None, None, :]  # [nS, nA, nS]
    # sum over a,s': [nS, d]
    part1 = np.einsum("ijk,ijkl->il", ew_h, step_feat)
    # Second part: sum_a sum_s' ew[s,a,s'] * U[s',dim] = full_M[s,s'] @ U[s',dim]
    part2 = full_M @ U  # [nS, d]

    rhs_U = (part1 + part2)[nonterm]
    U_non = solve(A_mat, rhs_U, assume_a="gen")
    U[nonterm] = U_non

    # V RHS for each (i,j):
    # sum_a sum_s' ew * (V[s',i,j] + psi_i*U[s',j] + psi_j*U[s',i] + psi_i*psi_j*h[s'])
    for i_dim in range(d):
        for j_dim in range(i_dim, d):
            # V[s',i,j] contribution: full_M @ V[:,i,j]
            t1 = full_M @ V[:, i_dim, j_dim]
            # psi_i * U[s',j]: sum_a sum_s' ew * psi_i * U[s',j]
            t2 = np.einsum("ijk,ijk,k->i", exp_weights, step_feat[:, :, :, i_dim], U[:, j_dim])
            # psi_j * U[s',i]
            t3 = np.einsum("ijk,ijk,k->i", exp_weights, step_feat[:, :, :, j_dim], U[:, i_dim])
            # psi_i * psi_j * h[s']
            t4 = np.einsum("ijk,ijk,ijk,k->i", exp_weights,
                           step_feat[:, :, :, i_dim], step_feat[:, :, :, j_dim], h)

            rhs = (t1 + t2 + t3 + t4)[nonterm]
            V_col = solve(A_mat, rhs, assume_a="gen")
            V[nonterm, i_dim, j_dim] = V_col
            if i_dim != j_dim:
                V[nonterm, j_dim, i_dim] = V_col

    h_safe = np.maximum(h, 1e-300)
    grad = U / h_safe[:, None]
    hess = V / h_safe[:, None, None] - grad[:, :, None] * grad[:, None, :]
    return grad, hess
