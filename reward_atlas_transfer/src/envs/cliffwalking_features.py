"""CliffWalking feature map: [goal_reached, cliff_fall, step_cost]."""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec


def build_cliffwalking_reward(model: TabularEpisodicModel) -> RewardSpec:
    """Build CliffWalking features. Dimension d=3.

    Layout: 4 rows × 12 cols. Start = bottom-left (state 36).
    Goal = bottom-right (state 47). Cliff = bottom row cols 1..10.
    """
    nS, nA = model.nS, model.nA
    d = 3
    ncols = 12
    nrows = 4

    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    step_feat[..., 2] = 1.0

    goal_state = nrows * ncols - 1
    term_feat[goal_state, 0] = 1.0

    for col in range(1, ncols - 1):
        s = (nrows - 1) * ncols + col
        if s < nS:
            term_feat[s, 1] = 1.0

    return RewardSpec(dim=d, step_features=step_feat, terminal_features=term_feat)
