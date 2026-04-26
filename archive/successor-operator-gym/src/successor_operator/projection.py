"""Policy projection and geometry-based zero-shot approximation.

Implements:
- projected_policy: KL-optimal projection of the tilted path law onto realizable policies
- gold_standard_tilted_law: the full joint q_w(a, s' | s)
- approximate_h_from_geometry: first/second-order Taylor expansion of log h
"""

import numpy as np

from successor_operator.mdp import TabularMDP


def projected_policy(mdp: TabularMDP, pi0: np.ndarray, step_feat: np.ndarray,
                     w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute the KL-projected policy pi_w^proj(a|s).

    This is the closed-form solution to the local joint-KL minimization:
        pi_w^proj(a|s) = pi0(a|s) * Z_w(s,a) / h_w(s)

    Vectorized implementation.

    Returns:
        pi: shape [nS, nA], projected policy.
    """
    nS, nA = mdp.nS, mdp.nA

    # Z_w(s,a) = sum_s' P(s'|s,a) * exp(psi(s,a,s') . w) * h(s')
    dot_products = np.einsum("ijkl,l->ijk", step_feat, w)
    dot_products = np.clip(dot_products, -500, 500)
    h_safe = np.where(np.isfinite(h), h, 0.0)
    z_sa = np.sum(mdp.P * np.exp(dot_products) * h_safe[None, None, :], axis=2)  # [nS, nA]

    pi = pi0 * z_sa
    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    pi = pi / row_sums
    pi = np.where(np.isfinite(pi), pi, pi0)

    pi[mdp.terminal] = 0.0
    return pi


def gold_standard_tilted_law(mdp: TabularMDP, pi0: np.ndarray, step_feat: np.ndarray,
                              w: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute the gold-standard tilted joint law q_w(a, s' | s).

    q_w(a,s'|s) = pi0(a|s) * P(s'|s,a) * exp(psi . w) * h(s') / h(s)

    Returns:
        q: shape [nS, nA, nS], the tilted one-step law.
    """
    nS = mdp.nS
    dot_products = np.einsum("ijkl,l->ijk", step_feat, w)
    q = pi0[:, :, None] * mdp.P * np.exp(dot_products) * h[None, None, :]

    h_safe = np.maximum(h, 1e-300)
    q = q / h_safe[:, None, None]
    q[mdp.terminal] = 0.0
    return q


def approximate_h_from_geometry(A_w: np.ndarray, grad: np.ndarray,
                                 hess: np.ndarray, delta: np.ndarray,
                                 order: int = 2) -> np.ndarray:
    """Approximate h_{w+delta} using Taylor expansion of log h_w.

    Args:
        A_w: log h_w, shape [nS].
        grad: gradient of log h_w, shape [nS, d].
        hess: Hessian of log h_w, shape [nS, d, d].
        delta: weight perturbation, shape [d].
        order: 1 for linear, 2 for quadratic approximation.

    Returns:
        h_approx: shape [nS], approximation of h_{w+delta}.
    """
    approx = A_w + grad @ delta
    if order >= 2:
        quad = np.einsum("i,sij,j->s", delta, hess, delta)
        approx = approx + 0.5 * quad
    approx = np.clip(approx, -500, 500)
    return np.exp(approx)
