"""Tests for Gymnasium model extraction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from successor_operator.gym_extract import extract_tabular_mdp, get_env_desc
from successor_operator.passive_policy import uniform_passive_policy


class TestExtractTabularMDP:
    def test_frozenlake_deterministic(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="4x4")
        assert mdp.nS == 16
        assert mdp.nA == 4
        assert mdp.P.shape == (16, 4, 16)
        assert mdp.terminal.sum() > 0
        mdp.validate()

    def test_frozenlake_slippery(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=True, map_name="4x4")
        assert mdp.nS == 16
        mdp.validate()

    def test_frozenlake_8x8(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="8x8")
        assert mdp.nS == 64
        mdp.validate()

    def test_cliffwalking(self):
        mdp = extract_tabular_mdp("CliffWalking-v1")
        assert mdp.nS == 48
        assert mdp.nA == 4
        mdp.validate()

    def test_taxi(self):
        mdp = extract_tabular_mdp("Taxi-v3")
        assert mdp.nS == 500
        assert mdp.nA == 6
        mdp.validate()

    def test_transition_rows_sum_to_one(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=True, map_name="4x4")
        for s in range(mdp.nS):
            if mdp.terminal[s]:
                continue
            for a in range(mdp.nA):
                if mdp.legal[s, a]:
                    assert abs(mdp.P[s, a].sum() - 1.0) < 1e-12

    def test_start_distribution(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="4x4")
        assert abs(mdp.start.sum() - 1.0) < 1e-12
        assert mdp.start[0] == 1.0

    def test_env_desc(self):
        desc = get_env_desc("FrozenLake-v1", is_slippery=False, map_name="4x4")
        assert desc is not None
        assert desc.shape == (4, 4)


class TestPassivePolicy:
    def test_uniform_sums_to_one(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="4x4")
        pi0 = uniform_passive_policy(mdp)
        for s in range(mdp.nS):
            if not mdp.terminal[s]:
                assert abs(pi0[s].sum() - 1.0) < 1e-12

    def test_uniform_only_legal(self):
        mdp = extract_tabular_mdp("FrozenLake-v1", is_slippery=False, map_name="4x4")
        pi0 = uniform_passive_policy(mdp)
        assert np.all(pi0[~mdp.legal] == 0.0)
