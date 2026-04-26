"""Evaluation metrics for successor operator experiments.

Terminal KL, occupancy KL, expected score gap, realizability gap, geometry error.
"""

import numpy as np

from successor_operator.mdp import TabularMDP


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> float:
    """KL(p || q) for discrete distributions. Handles zeros gracefully."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def terminal_kl(target_term_dist: np.ndarray, realized_term_dist: np.ndarray) -> float:
    """KL divergence between target and realized terminal distributions."""
    return kl_divergence(target_term_dist, realized_term_dist)


def occupancy_kl(target_occ: np.ndarray, realized_occ: np.ndarray) -> float:
    """KL between normalized occupancy measures."""
    t_sum = target_occ.sum()
    r_sum = realized_occ.sum()
    if t_sum == 0 or r_sum == 0:
        return float("inf")
    return kl_divergence(target_occ / t_sum, realized_occ / r_sum)


def expected_score_gap(
    mdp: TabularMDP,
    step_feat: np.ndarray,
    term_feat: np.ndarray,
    w: np.ndarray,
    target_occ: np.ndarray,
    target_term_dist: np.ndarray,
    realized_occ: np.ndarray,
    realized_term_dist: np.ndarray,
    pi_target: np.ndarray,
    pi_realized: np.ndarray,
) -> float:
    """Compute E_target[G_w] - E_realized[G_w].

    Uses occupancy + terminal distributions for exact computation.
    """
    def expected_score(occ, term_dist, pi):
        psi_dot_w = np.einsum("ijkl,l->ijk", step_feat, w)  # [nS, nA, nS]
        # sum_s occ[s] * sum_a pi[s,a] * sum_s' P[s,a,s'] * (psi . w)
        weighted = np.einsum("i,ia,iaj,iaj->", occ, pi, mdp.P, psi_dot_w)
        term_score = np.dot(term_dist, term_feat @ w)
        return weighted + term_score

    e_target = expected_score(target_occ, target_term_dist, pi_target)
    e_realized = expected_score(realized_occ, realized_term_dist, pi_realized)
    return float(e_target - e_realized)


def realizability_gap(
    mdp: TabularMDP,
    q_tilted: np.ndarray,
    pi_proj: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """Per-state realizability gap: KL(q_w(a,s'|s) || pi_proj(a|s) P(s'|s,a)).

    Args:
        q_tilted: shape [nS, nA, nS], the gold-standard tilted joint law.
        pi_proj: shape [nS, nA], the projected policy.

    Returns:
        gap: shape [nS], per-state KL divergence.
    """
    nS = mdp.nS
    gap = np.zeros(nS, dtype=np.float64)

    realizable = pi_proj[:, :, None] * mdp.P  # [nS, nA, nS]

    for s in range(nS):
        if mdp.terminal[s]:
            continue
        q_flat = q_tilted[s].ravel()
        r_flat = realizable[s].ravel()
        mask = q_flat > eps
        if not mask.any():
            continue
        gap[s] = np.sum(q_flat[mask] * np.log(q_flat[mask] / np.maximum(r_flat[mask], eps)))

    return gap


def geometry_error(h_exact: np.ndarray, h_approx: np.ndarray) -> dict:
    """Compute various error metrics between exact and approximate h.

    Returns dict with l2, linf, and relative errors on log h.
    """
    log_exact = np.log(np.maximum(h_exact, 1e-300))
    log_approx = np.log(np.maximum(h_approx, 1e-300))
    diff = log_exact - log_approx
    return {
        "l2_log": float(np.linalg.norm(diff)),
        "linf_log": float(np.max(np.abs(diff))),
        "rmse_log": float(np.sqrt(np.mean(diff ** 2))),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(log_exact), 1e-300)),
    }
