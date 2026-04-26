"""Evaluation metrics: KL divergences, geometry errors, score gaps."""

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def terminal_kl(target_td: np.ndarray, approx_td: np.ndarray) -> float:
    return kl_divergence(target_td, approx_td)


def occupancy_kl(target_occ: np.ndarray, approx_occ: np.ndarray) -> float:
    """KL between normalized occupancy measures (summed over time)."""
    t_flat = target_occ.ravel()
    a_flat = approx_occ.ravel()
    t_sum = t_flat.sum()
    a_sum = a_flat.sum()
    if t_sum == 0 or a_sum == 0:
        return float("inf")
    return kl_divergence(t_flat / t_sum, a_flat / a_sum)


def policy_kl(pi_target: np.ndarray, pi_approx: np.ndarray,
              model=None, eps: float = 1e-30) -> float:
    """Average per-(t,s) KL between target and approximate policies."""
    H, nS, nA = pi_target.shape
    total_kl = 0.0
    count = 0
    for t in range(H):
        for s in range(nS):
            if model is not None and model.terminal[s]:
                continue
            p = pi_target[t, s]
            q = pi_approx[t, s]
            mask = p > eps
            if mask.any():
                total_kl += np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps)))
            count += 1
    return float(total_kl / max(count, 1))


def geometry_error(h_exact: np.ndarray, h_approx: np.ndarray) -> dict:
    """Error metrics between exact and approximate h (in log space)."""
    log_ex = np.log(np.maximum(h_exact, 1e-300))
    log_ap = np.log(np.maximum(h_approx, 1e-300))
    diff = log_ex - log_ap
    finite_diff = diff[np.isfinite(diff)]
    if len(finite_diff) == 0:
        return {"rmse_log": float("inf"), "linf_log": float("inf"),
                "relative_l2": float("inf")}
    return {
        "rmse_log": float(np.sqrt(np.mean(finite_diff ** 2))),
        "linf_log": float(np.max(np.abs(finite_diff))),
        "relative_l2": float(
            np.linalg.norm(finite_diff) /
            max(np.linalg.norm(log_ex[np.isfinite(log_ex)]), 1e-300)
        ),
    }
