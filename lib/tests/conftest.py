"""Shared test fixtures for v2 codebase."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.envs.build_tabular_model import extract_tabular_model, uniform_policy
from src.envs.frozenlake_features import build_frozenlake_reward


@pytest.fixture
def fl_det_4x4():
    """FrozenLake deterministic 4x4 with H=30."""
    model = extract_tabular_model("FrozenLake-v1", H=30, is_slippery=False, map_name="4x4")
    reward = build_frozenlake_reward(model, {"is_slippery": False, "map_name": "4x4"})
    pi0 = uniform_policy(model)
    return model, reward, pi0


@pytest.fixture
def fl_slip_4x4():
    """FrozenLake slippery 4x4 with H=30."""
    model = extract_tabular_model("FrozenLake-v1", H=30, is_slippery=True, map_name="4x4")
    reward = build_frozenlake_reward(model, {"is_slippery": True, "map_name": "4x4"})
    pi0 = uniform_policy(model)
    return model, reward, pi0
