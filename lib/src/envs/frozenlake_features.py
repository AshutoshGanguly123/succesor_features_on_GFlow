"""FrozenLake feature map: [goal_reached, hole_hit, step_cost]."""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec, get_env_desc


def build_frozenlake_reward(model: TabularEpisodicModel,
                            env_kwargs: dict) -> RewardSpec:
    """Build FrozenLake features. Dimension d=3.

    Features: [goal_reached, hole_hit, step_cost].
    step_cost is 1.0 per step (weight should be negative).
    """
    nS, nA = model.nS, model.nA
    d = 3

    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 2] = 1.0

    desc = get_env_desc("FrozenLake-v1", **env_kwargs)
    if desc is not None:
        flat = desc.flatten()
        for s in range(min(nS, len(flat))):
            c = chr(flat[s]) if isinstance(flat[s], (np.uint8, int)) else str(flat[s])
            if c == "G":
                term_feat[s, 0] = 1.0
            elif c == "H":
                term_feat[s, 1] = 1.0

    return RewardSpec(dim=d, step_features=step_feat, terminal_features=term_feat)
