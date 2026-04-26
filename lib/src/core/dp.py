"""Exact backward dynamic programming for successor mass h_r(s,t).

Time-augmented finite-horizon solver: sweeps backward from t=H to t=0.
Vectorized per time-slice using numpy broadcasting.
"""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec


def solve_h(model: TabularEpisodicModel,
            reward: RewardSpec,
            pi0: np.ndarray,
            r: np.ndarray) -> np.ndarray:
    """Compute exact successor mass h_r(s,t) by backward DP.

    Args:
        model: TabularEpisodicModel.
        reward: RewardSpec with step/terminal features.
        pi0: Reference policy, shape [H, nS, nA].
        r: Reward weight vector, shape [d].

    Returns:
        h: shape [H+1, nS], successor mass at each (t, s).
    """
    H, nS, nA = model.H, model.nS, model.nA
    h = np.zeros((H + 1, nS), dtype=np.float64)

    tf_H = reward.get_term_feat(H)
    h[H] = np.exp(tf_H @ r)

    for t in range(H - 1, -1, -1):
        tf_t = reward.get_term_feat(t)
        sf_t = reward.get_step_feat(t)

        # Terminal states at this time step
        h[t, model.terminal] = np.exp(tf_t[model.terminal] @ r)

        # Nonterminal states: vectorized
        nt = model.nonterminal_mask()
        if not nt.any():
            continue

        # dot_products[s, a, s'] = sf_t[s, a, s'] . r
        dots = np.einsum("ijkl,l->ijk", sf_t, r)
        dots = np.clip(dots, -500, 500)

        # weighted[s, a, s'] = pi0[t,s,a] * P[s,a,s'] * exp(dots) * h[t+1,s']
        weighted = pi0[t, :, :, None] * model.P * np.exp(dots) * h[t + 1][None, None, :]

        # h[t, s] = sum_a sum_s' weighted[s, a, s']
        h_slice = weighted.sum(axis=(1, 2))
        h[t, nt] = h_slice[nt]

    return h


def solve_h_log_stable(model: TabularEpisodicModel,
                       reward: RewardSpec,
                       pi0: np.ndarray,
                       r: np.ndarray) -> np.ndarray:
    """Log-space stable backward DP using logsumexp.

    Slower but numerically stable for extreme reward weights.
    """
    from scipy.special import logsumexp as _logsumexp

    H, nS, nA = model.H, model.nS, model.nA
    log_h = np.full((H + 1, nS), -np.inf, dtype=np.float64)

    tf_H = reward.get_term_feat(H)
    log_h[H] = tf_H @ r

    for t in range(H - 1, -1, -1):
        tf_t = reward.get_term_feat(t)
        sf_t = reward.get_step_feat(t)

        log_h[t, model.terminal] = tf_t[model.terminal] @ r

        for s in model.nonterminal_indices():
            terms = []
            for a in range(nA):
                log_pi = np.log(max(pi0[t, s, a], 1e-300))
                for sp in range(nS):
                    p = model.P[s, a, sp]
                    if p == 0.0:
                        continue
                    val = log_pi + np.log(p) + (sf_t[s, a, sp] @ r) + log_h[t + 1, sp]
                    terms.append(val)
            if terms:
                log_h[t, s] = _logsumexp(terms)

    return np.exp(np.clip(log_h, -500, 500))
