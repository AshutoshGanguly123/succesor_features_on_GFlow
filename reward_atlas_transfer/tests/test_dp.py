"""Tests for backward DP solver."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.core.dp import solve_h
from src.core.bridge_projection import tilted_joint_law


class TestSolveH:
    def test_h_positive(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        assert h.shape == (model.H + 1, model.nS)
        assert np.all(h >= 0)

    def test_boundary_correct(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        tf = reward.get_term_feat(model.H)
        for s in range(model.nS):
            expected = np.exp(tf[s] @ w)
            assert abs(h[model.H, s] - expected) < 1e-12

    def test_zero_weights(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.zeros(3)
        h = solve_h(model, reward, pi0, w)
        for s in model.terminal_indices():
            assert abs(h[model.H, s] - 1.0) < 1e-12

    def test_tilted_law_normalizes(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        q = tilted_joint_law(model, reward, pi0, w, h)
        for t in range(model.H):
            for s in model.nonterminal_indices():
                total = q[t, s].sum()
                assert abs(total - 1.0) < 1e-8, f"t={t}, s={s}: sum={total}"

    def test_slippery(self, fl_slip_4x4):
        model, reward, pi0 = fl_slip_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        assert np.all(np.isfinite(h))

    def test_mc_consistency(self, fl_det_4x4):
        """h(s0, 0) matches MC estimate of E[exp(G)]."""
        model, reward, pi0 = fl_det_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_h(model, reward, pi0, w)

        rng = np.random.default_rng(42)
        n_samples = 30000
        exp_G = []
        for _ in range(n_samples):
            s = 0
            G = 0.0
            for t in range(model.H):
                if model.terminal[s]:
                    G += reward.get_term_feat(t)[s] @ w
                    break
                a = rng.choice(model.nA, p=pi0[t, s])
                sp = rng.choice(model.nS, p=model.P[s, a])
                G += reward.get_step_feat(t)[s, a, sp] @ w
                s = sp
            else:
                G += reward.get_term_feat(model.H)[s] @ w
            exp_G.append(np.exp(G))

        mc = np.mean(exp_G)
        rel_err = abs(mc - h[0, 0]) / max(abs(h[0, 0]), 1e-10)
        assert rel_err < 0.05, f"MC={mc:.4f} vs exact={h[0,0]:.4f}, rel={rel_err:.4f}"
