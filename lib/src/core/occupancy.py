"""State-time occupancy computation for finite-horizon episodic MDPs.

Computes d(s,t) = Pr(s_t = s) under a given time-dependent policy.
Also computes terminal distribution.
"""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel


def compute_occupancy(model: TabularEpisodicModel,
                      pi: np.ndarray) -> np.ndarray:
    """Compute state-time occupancy d(s,t) under policy pi.

    d(s,0) = rho0(s)
    d(s',t+1) = sum_s sum_a d(s,t) * pi(a|s,t) * P(s'|s,a)  [for nonterminal s]

    Args:
        model: Tabular episodic model.
        pi: Policy, shape [H, nS, nA].

    Returns:
        occ: shape [H+1, nS], occupancy at each (t, s).
    """
    H, nS = model.H, model.nS
    occ = np.zeros((H + 1, nS), dtype=np.float64)
    occ[0] = model.rho0.copy()

    # T[s, s'] = sum_a pi[t,s,a] * P[s,a,s']
    for t in range(H):
        pi_safe = np.where(np.isfinite(pi[t]), pi[t], 0.0)
        T = np.einsum("ia,iaj->ij", pi_safe, model.P)
        # Only propagate from nonterminal states
        d_nt = occ[t].copy()
        d_nt[model.terminal] = 0.0
        occ[t + 1] = d_nt @ T

    return occ


def compute_terminal_distribution(model: TabularEpisodicModel,
                                  pi: np.ndarray) -> np.ndarray:
    """Compute the terminal state distribution.

    Aggregates probability of termination at each state across all time steps.

    Returns:
        term_dist: shape [nS], probability of terminating at each state.
    """
    occ = compute_occupancy(model, pi)
    H, nS = model.H, model.nS
    term_dist = np.zeros(nS, dtype=np.float64)

    for t in range(H):
        pi_safe = np.where(np.isfinite(pi[t]), pi[t], 0.0)
        T = np.einsum("ia,iaj->ij", pi_safe, model.P)
        d_nt = occ[t].copy()
        d_nt[model.terminal] = 0.0
        flow_to_term = d_nt @ T
        term_dist[model.terminal] += flow_to_term[model.terminal]

    # At t=H, everything not yet absorbed is force-terminated
    term_dist += occ[H]

    total = term_dist.sum()
    if total > 0:
        term_dist = term_dist / total

    return term_dist


def compute_mean_episode_length(model: TabularEpisodicModel,
                                pi: np.ndarray) -> float:
    """Expected episode length under policy pi."""
    occ = compute_occupancy(model, pi)
    # Still-alive mass at each time
    alive = np.zeros(model.H + 1)
    for t in range(model.H + 1):
        alive[t] = occ[t, ~model.terminal].sum()

    # Expected length = sum_t alive[t] (probability of being active at step t)
    return float(alive[:model.H].sum())
