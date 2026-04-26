"""Shared test fixtures."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np

from successor_operator.mdp import TabularMDP
from successor_operator.gym_extract import extract_tabular_mdp, get_env_desc
from successor_operator.passive_policy import uniform_passive_policy
from successor_operator.features import frozenlake_features


@pytest.fixture
def fl_deterministic_4x4():
    """FrozenLake deterministic 4x4 fixture."""
    mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="4x4")
    desc = get_env_desc("FrozenLake-v1", is_slippery=False, map_name="4x4")
    pi0 = uniform_passive_policy(mdp)
    step_feat, term_feat = frozenlake_features(mdp, desc)
    return mdp, pi0, step_feat, term_feat


@pytest.fixture
def fl_slippery_4x4():
    """FrozenLake slippery 4x4 fixture."""
    mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=True, map_name="4x4")
    desc = get_env_desc("FrozenLake-v1", is_slippery=True, map_name="4x4")
    pi0 = uniform_passive_policy(mdp)
    step_feat, term_feat = frozenlake_features(mdp, desc)
    return mdp, pi0, step_feat, term_feat
