"""Tests for bridge projection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.core.dp import solve_h
from src.core.bridge_projection import project_policy, tilted_joint_law


class TestProjection:
    def test_sums_to_one(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        pi, _ = project_policy(model, reward, pi0, w, h)
        for t in range(model.H):
            for s in model.nonterminal_indices():
                assert abs(pi[t, s].sum() - 1.0) < 1e-10

    def test_matches_action_marginal(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        pi, _ = project_policy(model, reward, pi0, w, h)
        q = tilted_joint_law(model, reward, pi0, w, h)
        for t in range(model.H):
            for s in model.nonterminal_indices():
                action_marginal = q[t, s].sum(axis=1)
                np.testing.assert_allclose(pi[t, s], action_marginal, atol=1e-8)

    def test_nonneg(self, fl_slip_4x4):
        model, reward, pi0 = fl_slip_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        pi, _ = project_policy(model, reward, pi0, w, h)
        assert np.all(pi >= -1e-15)
