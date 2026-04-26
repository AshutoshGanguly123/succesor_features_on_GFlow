"""Tests for moment computation and geometry approximation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.core.dp import solve_h
from src.core.geometry import solve_moments, approx_h
from src.envs.build_tabular_model import Anchor


class TestMoments:
    def _finite_diff_grad(self, model, reward, pi0, w, eps=1e-5):
        h0 = solve_h(model, reward, pi0, w)
        A0 = np.log(np.maximum(h0, 1e-300))
        d = len(w)
        grad = np.zeros(h0.shape + (d,))
        for i in range(d):
            wp = w.copy(); wp[i] += eps
            wm = w.copy(); wm[i] -= eps
            Ap = np.log(np.maximum(solve_h(model, reward, pi0, wp), 1e-300))
            Am = np.log(np.maximum(solve_h(model, reward, pi0, wm), 1e-300))
            grad[..., i] = (Ap - Am) / (2 * eps)
        return grad

    def test_grad_matches_fd(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_h(model, reward, pi0, w)
        mu, Sigma = solve_moments(model, reward, pi0, w, h)
        grad_fd = self._finite_diff_grad(model, reward, pi0, w)

        for t in [0, 5, 10]:
            for s in model.nonterminal_indices()[:5]:
                np.testing.assert_allclose(
                    mu[t, s], grad_fd[t, s], atol=1e-4,
                    err_msg=f"Grad mismatch at t={t}, s={s}"
                )

    def test_hessian_symmetric(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([3.0, -3.0, -0.05])
        h = solve_h(model, reward, pi0, w)
        mu, Sigma = solve_moments(model, reward, pi0, w, h)
        for t in [0, 5]:
            for s in model.nonterminal_indices()[:5]:
                np.testing.assert_allclose(
                    Sigma[t, s], Sigma[t, s].T, atol=1e-10,
                    err_msg=f"Hessian not symmetric at t={t}, s={s}"
                )

    def test_approx_h_exact_at_anchor(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        w = np.array([5.0, -5.0, -0.1])
        h = solve_h(model, reward, pi0, w)
        mu, Sigma = solve_moments(model, reward, pi0, w, h)
        log_h = np.log(np.maximum(h, 1e-300))

        anchor = Anchor(name="test", r=w, h=h, log_h=log_h, mu=mu,
                        Sigma=Sigma, policy=np.zeros((model.H, model.nS, model.nA)),
                        metric=np.eye(3))

        h_approx = approx_h(anchor, w, order=2)
        np.testing.assert_allclose(h, h_approx, rtol=1e-8)
