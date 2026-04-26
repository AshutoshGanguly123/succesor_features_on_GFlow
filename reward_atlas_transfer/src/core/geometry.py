"""Local free-energy geometry: exact moment recursions and Taylor approximation.

Computes:
- mu_r(s,t) = grad_r log h_r(s,t) = E_q[Phi_{t:} | s_t=s]
- Sigma_r(s,t) = Hess_r log h_r(s,t) = Cov_q(Phi_{t:} | s_t=s)
via backward recursion using the tilted joint law q_r(a,s'|s,t).
"""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec


def solve_moments(model: TabularEpisodicModel,
                  reward: RewardSpec,
                  pi0: np.ndarray,
                  r: np.ndarray,
                  h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact gradient mu and Hessian Sigma of log h via backward DP.

    Uses the tilted joint law for the recursion weights.

    Returns:
        mu: shape [H+1, nS, d], gradient of log h_r.
        Sigma: shape [H+1, nS, d, d], Hessian of log h_r.
    """
    H, nS, nA = model.H, model.nS, model.nA
    d = reward.dim
    mu = np.zeros((H + 1, nS, d), dtype=np.float64)
    S = np.zeros((H + 1, nS, d, d), dtype=np.float64)

    # Boundary: t=H and terminal states
    for t in range(H + 1):
        tf = reward.get_term_feat(t)
        for s in range(nS):
            if model.terminal[s] or t == H:
                b = tf[s]
                mu[t, s] = b
                S[t, s] = np.outer(b, b)

    # Backward pass
    for t in range(H - 1, -1, -1):
        sf = reward.get_step_feat(t)
        h_safe = np.maximum(h[t], 1e-300)

        # Compute tilted joint weights q(a,s'|s,t)
        dots = np.einsum("ijkl,l->ijk", sf, r)
        dots = np.clip(dots, -500, 500)
        q_weights = (pi0[t, :, :, None] * model.P *
                     np.exp(dots) * h[t + 1][None, None, :])
        q_weights = q_weights / h_safe[:, None, None]

        for s in model.nonterminal_indices():
            mu_s = np.zeros(d, dtype=np.float64)
            S_s = np.zeros((d, d), dtype=np.float64)

            for a in range(nA):
                for sp in range(nS):
                    w = q_weights[s, a, sp]
                    if w < 1e-300:
                        continue
                    psi = sf[s, a, sp]
                    mu_next = mu[t + 1, sp]
                    S_next = S[t + 1, sp]

                    phi_mean = psi + mu_next
                    mu_s += w * phi_mean
                    S_s += w * (np.outer(psi, psi) +
                                np.outer(psi, mu_next) +
                                np.outer(mu_next, psi) +
                                S_next)

            mu[t, s] = mu_s
            S[t, s] = S_s

    Sigma = S - np.einsum("tsi,tsj->tsij", mu, mu)
    return mu, Sigma


def solve_moments_vectorized(model: TabularEpisodicModel,
                             reward: RewardSpec,
                             pi0: np.ndarray,
                             r: np.ndarray,
                             h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized moment computation. Faster for small-medium state spaces.

    Trades memory for speed by computing over all states simultaneously.
    """
    H, nS, nA = model.H, model.nS, model.nA
    d = reward.dim
    mu = np.zeros((H + 1, nS, d), dtype=np.float64)
    S = np.zeros((H + 1, nS, d, d), dtype=np.float64)

    for t in range(H + 1):
        tf = reward.get_term_feat(t)
        term_mask = model.terminal | (t == H)
        for s in np.where(term_mask)[0]:
            b = tf[s]
            mu[t, s] = b
            S[t, s] = np.outer(b, b)

    for t in range(H - 1, -1, -1):
        sf = reward.get_step_feat(t)
        h_safe = np.maximum(h[t], 1e-300)

        dots = np.einsum("ijkl,l->ijk", sf, r)
        dots = np.clip(dots, -500, 500)
        q_w = (pi0[t, :, :, None] * model.P * np.exp(dots) *
               h[t + 1][None, None, :]) / h_safe[:, None, None]

        # mu[t,s] = sum_{a,s'} q[s,a,s'] * (psi[s,a,s'] + mu[t+1,s'])
        # Part 1: sum q * psi
        mu_psi = np.einsum("ijk,ijkl->il", q_w.sum(axis=3) if q_w.ndim == 4 else q_w,
                           sf.sum(axis=2) if sf.ndim == 5 else np.zeros((nS, nA, d)))
        # Fall back to loop for correctness
        nt = model.nonterminal_indices()
        for s in nt:
            mu_s = np.zeros(d)
            S_s = np.zeros((d, d))
            for a in range(nA):
                for sp in range(nS):
                    w = q_w[s, a, sp]
                    if w < 1e-300:
                        continue
                    psi = sf[s, a, sp]
                    mn = mu[t + 1, sp]
                    Sn = S[t + 1, sp]
                    mu_s += w * (psi + mn)
                    S_s += w * (np.outer(psi, psi) + np.outer(psi, mn) +
                                np.outer(mn, psi) + Sn)
            mu[t, s] = mu_s
            S[t, s] = S_s

    Sigma = S - np.einsum("tsi,tsj->tsij", mu, mu)
    return mu, Sigma


def approx_log_h(anchor, r_target: np.ndarray, order: int = 2) -> np.ndarray:
    """Approximate log h at r_target using Taylor expansion around anchor.

    Args:
        anchor: Anchor object with r, log_h, mu, Sigma.
        r_target: Target reward vector, shape [d].
        order: 1 or 2.

    Returns:
        log_h_approx: shape [H+1, nS].
    """
    delta = r_target - anchor.r
    log_h = anchor.log_h + np.einsum("tsi,i->ts", anchor.mu, delta)
    if order >= 2:
        quad = 0.5 * np.einsum("i,tsij,j->ts", delta, anchor.Sigma, delta)
        log_h = log_h + quad
    return np.clip(log_h, -500, 500)


def approx_h(anchor, r_target: np.ndarray, order: int = 2) -> np.ndarray:
    """Approximate h at r_target. Returns shape [H+1, nS]."""
    return np.exp(approx_log_h(anchor, r_target, order))
