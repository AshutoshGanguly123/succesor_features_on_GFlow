"""Tests for atlas and continuation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.core.atlas import build_anchor, build_atlas, select_anchor, anchor_distance
from src.core.occupancy import compute_terminal_distribution, compute_occupancy


class TestAtlas:
    def test_build_anchor(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        r = np.array([5.0, -5.0, -0.1])
        anchor = build_anchor("test", r, model, reward, pi0)
        assert anchor.h.shape == (model.H + 1, model.nS)
        assert anchor.mu.shape == (model.H + 1, model.nS, 3)
        assert anchor.Sigma.shape == (model.H + 1, model.nS, 3, 3)
        assert anchor.policy.shape == (model.H, model.nS, model.nA)

    def test_build_atlas(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        specs = [
            {"name": "a1", "r": [5.0, -5.0, -0.1]},
            {"name": "a2", "r": [10.0, -3.0, -0.1]},
        ]
        atlas = build_atlas(specs, model, reward, pi0)
        assert len(atlas) == 2

    def test_select_anchor(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        specs = [
            {"name": "standard", "r": [5.0, -5.0, -0.1]},
            {"name": "goal_heavy", "r": [10.0, -3.0, -0.1]},
        ]
        atlas = build_atlas(specs, model, reward, pi0)
        r_target = np.array([5.5, -4.5, -0.1])
        best, scored = select_anchor(atlas, r_target)
        assert best.name == "standard"

    def test_anchor_distance_zero_at_self(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        r = np.array([5.0, -5.0, -0.1])
        anchor = build_anchor("test", r, model, reward, pi0)
        d = anchor_distance(anchor, r)
        assert d < 1e-8

    def test_occupancy_sums_to_one_ish(self, fl_det_4x4):
        model, reward, pi0 = fl_det_4x4
        r = np.array([5.0, -5.0, -0.1])
        anchor = build_anchor("test", r, model, reward, pi0)
        td = compute_terminal_distribution(model, anchor.policy)
        assert abs(td.sum() - 1.0) < 1e-6
