"""Tests for moment computation (gradient and Hessian of log h)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from successor_operator.feynman_kac import solve_successor_mass
from successor_operator.moments import solve_moments, solve_moments_vectorized


class TestMoments:
    def _finite_diff_grad(self, mdp, pi0, step_feat, term_feat, w, eps=1e-5):
        """Finite-difference gradient of log h_w."""
        h0 = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        A0 = np.log(np.maximum(h0, 1e-300))
        d = len(w)
        grad_fd = np.zeros((mdp.nS, d))
        for i in range(d):
            wp = w.copy()
            wp[i] += eps
            hp = solve_successor_mass(mdp, pi0, step_feat, term_feat, wp)
            Ap = np.log(np.maximum(hp, 1e-300))

            wm = w.copy()
            wm[i] -= eps
            hm = solve_successor_mass(mdp, pi0, step_feat, term_feat, wm)
            Am = np.log(np.maximum(hm, 1e-300))

            grad_fd[:, i] = (Ap - Am) / (2 * eps)
        return grad_fd

    def _finite_diff_hess(self, mdp, pi0, step_feat, term_feat, w, eps=1e-4):
        """Finite-difference Hessian of log h_w."""
        d = len(w)
        hess_fd = np.zeros((mdp.nS, d, d))
        for i in range(d):
            for j in range(d):
                wpp = w.copy(); wpp[i] += eps; wpp[j] += eps
                wpm = w.copy(); wpm[i] += eps; wpm[j] -= eps
                wmp = w.copy(); wmp[i] -= eps; wmp[j] += eps
                wmm = w.copy(); wmm[i] -= eps; wmm[j] -= eps

                hpp = np.log(np.maximum(solve_successor_mass(mdp, pi0, step_feat, term_feat, wpp), 1e-300))
                hpm = np.log(np.maximum(solve_successor_mass(mdp, pi0, step_feat, term_feat, wpm), 1e-300))
                hmp = np.log(np.maximum(solve_successor_mass(mdp, pi0, step_feat, term_feat, wmp), 1e-300))
                hmm = np.log(np.maximum(solve_successor_mass(mdp, pi0, step_feat, term_feat, wmm), 1e-300))

                hess_fd[:, i, j] = (hpp - hpm - hmp + hmm) / (4 * eps * eps)
        return hess_fd

    def test_gradient_matches_finite_diff(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        grad, _hess = solve_moments(mdp, pi0, step_feat, term_feat, w, h)
        grad_fd = self._finite_diff_grad(mdp, pi0, step_feat, term_feat, w)

        for s in mdp.nonterminal_indices():
            np.testing.assert_allclose(grad[s], grad_fd[s], atol=1e-5,
                                       err_msg=f"Gradient mismatch at state {s}")

    def test_hessian_matches_finite_diff(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        _grad, hess = solve_moments(mdp, pi0, step_feat, term_feat, w, h)
        hess_fd = self._finite_diff_hess(mdp, pi0, step_feat, term_feat, w)

        for s in mdp.nonterminal_indices():
            np.testing.assert_allclose(hess[s], hess_fd[s], atol=1e-4,
                                       err_msg=f"Hessian mismatch at state {s}")

    def test_vectorized_matches_loop(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)

        grad_loop, hess_loop = solve_moments(mdp, pi0, step_feat, term_feat, w, h)
        grad_vec, hess_vec = solve_moments_vectorized(mdp, pi0, step_feat, term_feat, w, h)

        np.testing.assert_allclose(grad_loop, grad_vec, atol=1e-10)
        np.testing.assert_allclose(hess_loop, hess_vec, atol=1e-10)

    def test_hessian_symmetric(self, fl_deterministic_4x4):
        mdp, pi0, step_feat, term_feat = fl_deterministic_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        _grad, hess = solve_moments(mdp, pi0, step_feat, term_feat, w, h)

        for s in mdp.nonterminal_indices():
            np.testing.assert_allclose(hess[s], hess[s].T, atol=1e-12,
                                       err_msg=f"Hessian not symmetric at state {s}")

    def test_slippery_grad(self, fl_slippery_4x4):
        mdp, pi0, step_feat, term_feat = fl_slippery_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w)
        grad, hess = solve_moments_vectorized(mdp, pi0, step_feat, term_feat, w, h)
        grad_fd = self._finite_diff_grad(mdp, pi0, step_feat, term_feat, w)

        for s in mdp.nonterminal_indices():
            np.testing.assert_allclose(grad[s], grad_fd[s], atol=1e-4,
                                       err_msg=f"Gradient mismatch at state {s}")
