"""Extract tabular MDP from Gymnasium environments."""

import gymnasium as gym
import numpy as np

from successor_operator.mdp import TabularMDP


def extract_tabular_mdp(env_name: str, **kwargs) -> TabularMDP:
    """Extract a TabularMDP from a Gymnasium environment with a .P attribute.

    Works with FrozenLake-v1, CliffWalking-v1, Taxi-v3, and any other
    Gymnasium Toy Text environment that exposes env.unwrapped.P.
    """
    env = gym.make(env_name, **kwargs)
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise ValueError(f"{env_name} does not expose a tabular transition model")

    nS = base.observation_space.n
    nA = base.action_space.n
    P = np.zeros((nS, nA, nS), dtype=np.float64)
    terminal = np.zeros(nS, dtype=bool)
    legal = np.ones((nS, nA), dtype=bool)

    for s in range(nS):
        for a in range(nA):
            outcomes = base.P[s][a]
            if len(outcomes) == 0:
                legal[s, a] = False
                continue
            for prob, s_next, _reward, done in outcomes:
                P[s, a, s_next] += prob
                if done:
                    terminal[s_next] = True

    start = np.zeros(nS, dtype=np.float64)
    if hasattr(base, "initial_state_distrib"):
        start[:] = np.asarray(base.initial_state_distrib, dtype=np.float64)
    else:
        obs, _ = env.reset(seed=0)
        start[int(obs)] = 1.0

    env.close()

    mdp = TabularMDP(nS=nS, nA=nA, P=P, terminal=terminal, start=start, legal=legal)
    return mdp


def get_env_desc(env_name: str, **kwargs) -> np.ndarray:
    """Get the map description array for grid-based environments."""
    env = gym.make(env_name, **kwargs)
    base = env.unwrapped
    desc = None
    if hasattr(base, "desc"):
        desc = np.asarray(base.desc)
    env.close()
    return desc
