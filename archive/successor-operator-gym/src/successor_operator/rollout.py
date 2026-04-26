"""Rollout / episode sampling and model-based terminal distribution computation."""

import numpy as np
import gymnasium as gym

from successor_operator.mdp import TabularMDP


def sample_episode(env, pi: np.ndarray, max_steps: int = 1000,
                   seed: int = None, rng: np.random.Generator = None):
    """Sample a single episode from env using tabular policy pi.

    Args:
        env: Gymnasium environment instance.
        pi: Policy array, shape [nS, nA].
        max_steps: Maximum steps before truncation.
        seed: Random seed for env reset.
        rng: Numpy random generator for action sampling.

    Returns:
        traj: List of visited states (including terminal).
        actions: List of actions taken.
        done: Whether episode terminated naturally.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    obs, _info = env.reset(seed=seed)
    traj = [int(obs)]
    actions = []
    done = False
    t = 0
    while not done and t < max_steps:
        s = int(obs)
        probs = pi[s]
        if probs.sum() == 0:
            break
        a = rng.choice(len(probs), p=probs)
        obs2, _reward, terminated, truncated, _info = env.step(a)
        actions.append(a)
        traj.append(int(obs2))
        obs = obs2
        done = terminated or truncated
        t += 1
    return traj, actions, done


def sample_episodes_batch(env_name: str, pi: np.ndarray, n_episodes: int = 1000,
                          max_steps: int = 1000, seed: int = 42, **env_kwargs):
    """Sample multiple episodes and return trajectories.

    Returns:
        trajs: List of (traj, actions, done) tuples.
    """
    env = gym.make(env_name, **env_kwargs)
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_episodes):
        traj, actions, done = sample_episode(
            env, pi, max_steps=max_steps, seed=None, rng=rng
        )
        results.append((traj, actions, done))
    env.close()
    return results


def compute_terminal_distribution_model(mdp: TabularMDP, pi: np.ndarray,
                                         max_steps: int = 500) -> np.ndarray:
    """Compute the exact terminal state distribution via matrix powers.

    Uses the absorbing Markov chain induced by policy pi on the MDP.

    Returns:
        term_dist: shape [nS], probability of terminating in each state.
    """
    nS = mdp.nS
    nonterm = mdp.nonterminal_indices()
    term = mdp.terminal_indices()

    if len(nonterm) == 0:
        return mdp.start.copy()

    # T_full[i, j] = sum_a pi[i,a] * P[i,a,j]
    pi_safe = np.where(np.isfinite(pi), pi, 0.0)
    T_full = np.einsum("ia,iaj->ij", pi_safe, mdp.P)
    Q = T_full[np.ix_(nonterm, nonterm)]

    R = T_full[np.ix_(nonterm, term)]

    from scipy.linalg import solve
    A_mat = np.eye(len(nonterm)) - Q
    if not np.all(np.isfinite(A_mat)):
        term_dist = np.zeros(nS, dtype=np.float64)
        term_dist[term] = mdp.start[term]
        return term_dist
    N = solve(A_mat, np.eye(len(nonterm)), assume_a="gen")

    # Absorption probabilities B = N @ R, shape [n_nonterm, n_term]
    B = N @ R

    # Terminal distribution from the start distribution
    start_nonterm = mdp.start[nonterm]
    start_term = mdp.start[term]

    term_probs_from_nonterm = start_nonterm @ B  # [n_term]
    term_dist = np.zeros(nS, dtype=np.float64)
    term_dist[term] = term_probs_from_nonterm + start_term
    return term_dist


def compute_state_occupancy(mdp: TabularMDP, pi: np.ndarray, discount: float = 1.0,
                            max_steps: int = 500) -> np.ndarray:
    """Compute the (undiscounted) state occupancy measure under policy pi.

    For absorbing MDPs with discount=1, this is the expected number of visits.

    Returns:
        occ: shape [nS], expected visit count for each state.
    """
    nS = mdp.nS
    nonterm = mdp.nonterminal_indices()

    pi_safe = np.where(np.isfinite(pi), pi, 0.0)
    T_full = np.einsum("ia,iaj->ij", pi_safe, mdp.P)
    Q = T_full[np.ix_(nonterm, nonterm)]

    if discount < 1.0:
        Q = discount * Q

    from scipy.linalg import solve
    N = solve(np.eye(len(nonterm)) - Q, np.eye(len(nonterm)), assume_a="gen")

    occ = np.zeros(nS, dtype=np.float64)
    start_nonterm = mdp.start[nonterm]
    occ[nonterm] = start_nonterm @ N

    # Terminal states: reached from nonterminal states
    term = mdp.terminal_indices()
    if len(term) > 0:
        R = T_full[np.ix_(nonterm, term)]
        occ[term] = (start_nonterm @ N) @ R + mdp.start[term]

    return occ
