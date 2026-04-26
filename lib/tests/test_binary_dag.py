"""Tests for binary DAG experiment utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.envs.binary_dag import BinaryDAG


def test_binary_dag_size_depth_8():
    dag = BinaryDAG.from_random(depth=8, feature_dim=4, seed=1)
    assert dag.n_internal == 255
    assert dag.n_terminals == 256
    assert dag.n_nodes == 511


def test_descendant_measure_rows_sum_to_one():
    dag = BinaryDAG.from_random(depth=5, feature_dim=3, seed=2)
    w0 = np.array([0.4, -0.2, 0.1], dtype=np.float64)
    _, pi0 = dag.reference_flow_and_policy(w0)
    mu = dag.descendant_measure(pi0)
    row_sums = mu.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(dag.n_nodes), atol=1e-12)


def test_h_matches_manual_depth_2():
    # Four leaves in left-to-right order.
    feats = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    dag = BinaryDAG(depth=2, terminal_features=feats)

    # Uniform branching policy at internal nodes.
    pi = np.full((dag.n_internal, 2), 0.5, dtype=np.float64)
    mu = dag.descendant_measure(pi)

    w_linear = np.array([2.0, -1.0], dtype=np.float64)
    h_linear = dag.h_from_mu(mu, w_linear, reward_family="linear")

    # Root expectation over four leaves.
    leaf_linear = feats @ w_linear
    expected_root = leaf_linear.mean()
    assert abs(h_linear[0] - expected_root) < 1e-12

    w_log = np.array([0.3, -0.1], dtype=np.float64)
    h_log = dag.h_from_mu(mu, w_log, reward_family="loglinear")
    leaf_log = np.exp(feats @ w_log)
    expected_root_log = leaf_log.mean()
    assert abs(h_log[0] - expected_root_log) < 1e-12
