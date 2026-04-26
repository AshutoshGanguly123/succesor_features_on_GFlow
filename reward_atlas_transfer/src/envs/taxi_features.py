"""Taxi feature map: [successful_dropoff, illegal_action, step_cost, pickup_event]."""

import numpy as np
import gymnasium as gym
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec


def build_taxi_reward(model: TabularEpisodicModel) -> RewardSpec:
    """Build Taxi features. Dimension d=4.

    Features: [successful_dropoff, illegal_action, step_cost, pickup_event].
    Derived from the environment's native reward signal.
    """
    nS, nA = model.nS, model.nA
    d = 4

    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 2] = 1.0

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
                    step_feat[s, a, s_next, 1] = 1.0
                if a == 4 and reward > -8 and not done:
                    step_feat[s, a, s_next, 3] = 1.0

    env.close()
    return RewardSpec(dim=d, step_features=step_feat, terminal_features=term_feat)
