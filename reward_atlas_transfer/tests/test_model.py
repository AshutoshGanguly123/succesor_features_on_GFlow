"""Tests for tabular model extraction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.envs.build_tabular_model import extract_tabular_model, uniform_policy


class TestExtract:
    def test_frozenlake_det(self):
        m = extract_tabular_model("FrozenLake-v1", H=50, is_slippery=False, map_name="4x4")
        assert m.nS == 16 and m.nA == 4 and m.H == 50
        m.validate()

    def test_frozenlake_slip(self):
        m = extract_tabular_model("FrozenLake-v1", H=50, is_slippery=True, map_name="4x4")
        m.validate()

    def test_cliffwalking(self):
        m = extract_tabular_model("CliffWalking-v1", H=100)
        assert m.nS == 48 and m.nA == 4
        m.validate()

    def test_taxi(self):
        m = extract_tabular_model("Taxi-v3", H=20)
        assert m.nS == 500 and m.nA == 6
        m.validate()

    def test_start_dist(self):
        m = extract_tabular_model("FrozenLake-v1", H=30, is_slippery=False, map_name="4x4")
        assert abs(m.rho0.sum() - 1.0) < 1e-12

    def test_uniform_policy_shape(self):
        m = extract_tabular_model("FrozenLake-v1", H=20, is_slippery=False, map_name="4x4")
        pi = uniform_policy(m)
        assert pi.shape == (20, 16, 4)
        for t in range(20):
            for s in range(16):
                if not m.terminal[s]:
                    assert abs(pi[t, s].sum() - 1.0) < 1e-12
