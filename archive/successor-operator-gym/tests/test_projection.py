"""Tests for policy projection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from successor_operator.feynman_kac import solve_successor_mass
from successor_operator.projection import (
    projected_policy, gold_standard_tilted_law
)


class TestProjectedPolicy:
    def test_sums_to_one(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        pi = projected_policy(mdp, pi0, step_feat, w, h)

        for s in mdp.nonterminal_indices():
            assert abs(pi[s].sum() - 1.0) < 1e-12, f"State {s}: pi sums to {pi[s].sum()}"

    def test_matches_action_marginal(self, fl_deterministic_4x4):
        """pi_proj should equal the action marginal of q_w."""
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        pi = projected_policy(mdp, pi0, step_feat, w, h)
        q = gold_standard_tilted_law(mdp, pi0, step_feat, w, h)

        for s in mdp.nonterminal_indices():
            action_marginal = q[s].sum(axis=1)  # sum over s'
            np.testing.assert_allclose(pi[s], action_marginal, atol=1e-10)

    def test_deterministic_exact_match(self, fl_deterministic_4x4):
        """In deterministic env, projected policy should match exact tilt."""
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        pi = projected_policy(mdp, pi0, step_feat, w, h)
        q = gold_standard_tilted_law(mdp, pi0, step_feat, w, h)

        for s in mdp.nonterminal_indices():
            q_joint = q[s]
            pi_times_P = pi[s, :, None] * mdp.P[s]
            diff = np.abs(q_joint - pi_times_P).max()
            assert diff < 1e-10, f"State {s}: max diff {diff}"

    def test_nonneg(self, fl_slippery_4x4):
        mdp, pi0, step_feat, term_feat = fl_slippery_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        pi = projected_policy(mdp, pi0, step_feat, w, h)
        assert np.all(pi >= -1e-15)
