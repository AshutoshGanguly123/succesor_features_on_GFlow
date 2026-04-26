"""Feature functions for Gymnasium environments.

Each feature function returns:
    step_feat: shape [nS, nA, nS, d] - per-transition features
    term_feat: shape [nS, d] - terminal state features
where d is the feature dimension.
"""

import numpy as np

from successor_operator.mdp import TabularMDP


def frozenlake_features(mdp: TabularMDP, env_desc: np.ndarray):
    """FrozenLake features: [goal_reached, fell_in_hole, step_count].

    Feature dimension d=3.
    """
    nS, nA = mdp.nS, mdp.nA
    d = 3
    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 2] = 1.0

    flat = env_desc.flatten()
    for s in range(min(nS, len(flat))):
        c = chr(flat[s]) if isinstance(flat[s], (np.uint8, int)) else str(flat[s])
        if c == "G":
            term_feat[s, 0] = 1.0
        elif c == "H":
            term_feat[s, 1] = 1.0

    return step_feat, term_feat


def cliffwalking_features(mdp: TabularMDP, cliff_states: np.ndarray = None):
    """CliffWalking features: [goal_reached, cliff_hit, step_count].

    Feature dimension d=3.
    cliff_states should be a boolean array marking cliff cells. If None, we
    infer them from the environment structure (bottom row, columns 1..10).
    """
    nS, nA = mdp.nS, mdp.nA
    d = 3
    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 2] = 1.0

    ncols = 12
    nrows = 4
    goal_state = nrows * ncols - 1

    if cliff_states is None:
        cliff_states = np.zeros(nS, dtype=bool)
        for col in range(1, ncols - 1):
            s = (nrows - 1) * ncols + col
            if s < nS:
                cliff_states[s] = True

    term_feat[goal_state, 0] = 1.0

    for s in range(nS):
        if cliff_states[s]:
            term_feat[s, 1] = 1.0

    return step_feat, term_feat


def taxi_features(mdp: TabularMDP):
    """Taxi features: [successful_dropoff, illegal_action, pickup, step_count].

    Feature dimension d=4.
    We encode these via the reward structure of the environment:
    - successful dropoff: reward +20 transitions
    - illegal action: reward -10 transitions
    - normal step: reward -1
    """
    nS, nA = mdp.nS, mdp.nA
    d = 4
    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 3] = 1.0

    import gymnasium as gym
    env = gym.make("Taxi-v3")
    base = env.unwrapped

    for s in range(nS):
        for a in range(nA):
            for prob, s_next, reward, done in base.P[s][a]:
                if prob == 0.0:
                    continue
                if done and reward >= 10:
                    term_feat[s_next, 0] = 1.0
                if reward <= -8:
                    step_feat[s, a, s_next, 1] += 1.0
                if a == 4 and reward > -8 and not done:
                    step_feat[s, a, s_next, 2] += 1.0

    env.close()
    return step_feat, term_feat
