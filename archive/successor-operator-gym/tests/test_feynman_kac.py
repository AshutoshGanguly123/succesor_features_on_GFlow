"""Tests for the Feynman-Kac / successor mass solver."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from successor_operator.feynman_kac import solve_successor_mass
from successor_operator.projection import gold_standard_tilted_law


class TestSolveSuccessorMass:
    def test_h_positive(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        assert np.all(h >= 0)
        assert np.all(h[~mdp.terminal] > 0)

    def test_terminal_values(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        for s in mdp.terminal_indices():
            expected = np.exp(term_feat[s] @ w)
            assert abs(h[s] - expected) < 1e-12

    def test_zero_weights(self, fl_deterministic_4x4):
        """With w=0, h should be 1 at terminals and positive everywhere."""
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.zeros(3)
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        for s in mdp.terminal_indices():
            assert abs(h[s] - 1.0) < 1e-12

    def test_slippery(self, fl_slippery_4x4):
        mdp, pi0, step_feat, term_feat = fl_slippery_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        assert np.all(h >= 0)

    def test_tilted_law_normalizes(self, fl_deterministic_4x4):
        """q_w(a,s'|s) should sum to 1 for every nonterminal state."""
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        q = gold_standard_tilted_law(mdp, pi0, step_feat, w, h)

        for s in mdp.nonterminal_indices():
            total = q[s].sum()
            assert abs(total - 1.0) < 1e-10, f"State {s}: q sums to {total}"

    def test_monte_carlo_consistency(self, fl_deterministic_4x4):
        """h_w from linear system matches MC estimate of E[exp(G_w)] for start state."""
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)

        rng = np.random.default_rng(42)
        n_samples = 50000
        exp_G_samples = []

        for _ in range(n_samples):
            s = 0  # start state
            G = 0.0
            for _ in range(200):
                if mdp.terminal[s]:
                    G += term_feat[s] @ w
                    break
                probs = pi0[s]
                a = rng.choice(mdp.nA, p=probs)
                probs_next = mdp.P[s, a]
                s_next = rng.choice(mdp.nS, p=probs_next)
                G += step_feat[s, a, s_next] @ w
                s = s_next
            exp_G_samples.append(np.exp(G))

        mc_estimate = np.mean(exp_G_samples)
        rel_error = abs(mc_estimate - h[0]) / max(abs(h[0]), 1e-10)
        assert rel_error < 0.05, f"MC estimate {mc_estimate:.4f} vs exact {h[0]:.4f}, rel_error={rel_error:.4f}"
