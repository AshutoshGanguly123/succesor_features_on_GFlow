"""Bridge projection: project the ideal tilted law back to policy space.

Implements:
- project_policy: KL-optimal policy projection (closed form)
- tilted_joint_law: gold-standard q_r(a, s' | s, t)
"""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec


def compute_action_scores(model: TabularEpisodicModel,
                          reward: RewardSpec,
                          r: np.ndarray,
                          h: np.ndarray) -> np.ndarray:
    """Compute Z_r(s, a, t) = sum_s' P(s'|s,a) exp(psi.r) h(s',t+1).

    Returns:
        z_sa: shape [H, nS, nA].
    """
    H, nS, nA = model.H, model.nS, model.nA
    z_sa = np.zeros((H, nS, nA), dtype=np.float64)

    for t in range(H):
        sf = reward.get_step_feat(t)
        dots = np.einsum("ijkl,l->ijk", sf, r)
        dots = np.clip(dots, -500, 500)
        # z_sa[t, s, a] = sum_s' P[s,a,s'] * exp(dots[s,a,s']) * h[t+1,s']
        z_sa[t] = np.sum(model.P * np.exp(dots) * h[t + 1][None, None, :], axis=2)

    return z_sa


def project_policy(model: TabularEpisodicModel,
                   reward: RewardSpec,
                   pi0: np.ndarray,
                   r: np.ndarray,
                   h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the KL-projected policy pi_r^proj(a|s,t).

    pi_r^proj(a|s,t) = pi0(a|s,t) * Z_r(s,a,t) / h_r(s,t)

    Returns:
        pi: shape [H, nS, nA], projected policy.
        z_sa: shape [H, nS, nA], action scores.
    """
    H, nS, nA = model.H, model.nS, model.nA
    z_sa = compute_action_scores(model, reward, r, h)

    pi = pi0 * z_sa
    row_sums = pi.sum(axis=2, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    pi = pi / row_sums
    pi = np.where(np.isfinite(pi), pi, pi0)

    for t in range(H):
        pi[t, model.terminal] = 0.0

    return pi, z_sa


def tilted_joint_law(model: TabularEpisodicModel,
                     reward: RewardSpec,
                     pi0: np.ndarray,
                     r: np.ndarray,
                     h: np.ndarray) -> np.ndarray:
    """Compute the gold-standard tilted joint law q_r(a, s' | s, t).

    q_r(a,s'|s,t) = pi0(a|s,t) P(s'|s,a) exp(psi.r) h(s',t+1) / h(s,t)

    Returns:
        q: shape [H, nS, nA, nS].
    """
    H, nS, nA = model.H, model.nS, model.nA
    q = np.zeros((H, nS, nA, nS), dtype=np.float64)

    for t in range(H):
        sf = reward.get_step_feat(t)
        dots = np.einsum("ijkl,l->ijk", sf, r)
        dots = np.clip(dots, -500, 500)
        q_t = (pi0[t, :, :, None] * model.P * np.exp(dots) * h[t + 1][None, None, :])
        h_safe = np.maximum(h[t], 1e-300)
        q_t = q_t / h_safe[:, None, None]
        q_t[model.terminal] = 0.0
        q[t] = q_t

    return q
