"""Passive (reference) policy construction."""

import numpy as np

from successor_operator.mdp import TabularMDP


def uniform_passive_policy(mdp: TabularMDP) -> np.ndarray:
    """Uniform distribution over legal actions at each state.

    Returns:
        pi0: shape [nS, nA], with pi0[s] summing to 1 for nonterminal states.
    """
    pi0 = mdp.legal.astype(np.float64)
    row_sums = pi0.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return pi0 / row_sums
