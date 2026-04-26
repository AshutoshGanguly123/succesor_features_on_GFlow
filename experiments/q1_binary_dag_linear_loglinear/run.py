#!/usr/bin/env python3
"""Q1: linear exactness and global log-linear obstruction on a binary DAG."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reward_atlas_transfer"))

from src.envs.binary_dag import BinaryDAG, kl_divergence


def run_q1(
    depth: int = 8,
    feature_dim: int = 6,
    seed: int = 7,
    n_train_w: int = 64,
    n_eval_w: int = 24,
) -> dict:
    dag = BinaryDAG.from_random(depth=depth, feature_dim=feature_dim, seed=seed)
    rng = np.random.default_rng(seed + 11)
    w0 = rng.normal(size=feature_dim)
    _, pi0 = dag.reference_flow_and_policy(w0)
    mu = dag.descendant_measure(pi0)
    psi = dag.successor_features(mu)

    # Linear rewards: h = Psi^T w exactly.
    linear_errors = []
    for _ in range(n_eval_w):
        w = rng.normal(size=feature_dim)
        h_exact = dag.h_from_mu(mu, w, reward_family="linear")
        h_pred = np.einsum("sd,d->s", psi, w)
        linear_errors.append(np.max(np.abs(h_exact - h_pred)))
    linear_max_abs_error = float(np.max(linear_errors))
    linear_mean_abs_error = float(np.mean(linear_errors))

    # Log-linear: fit one global linear map log h ~= B^T w.
    X_train = rng.normal(size=(n_train_w, feature_dim))
    Y_train = np.stack(
        [dag.log_h_from_mu(mu, w, reward_family="loglinear") for w in X_train], axis=0
    )  # [n_train_w, n_nodes]
    B, *_ = np.linalg.lstsq(X_train, Y_train, rcond=None)  # [d, n_nodes]

    X_eval = rng.normal(size=(n_eval_w, feature_dim))
    rmse_log = []
    kl_terms = []
    for w in X_eval:
        y_true = dag.log_h_from_mu(mu, w, reward_family="loglinear")
        y_hat = w @ B
        rmse_log.append(float(np.sqrt(np.mean((y_hat - y_true) ** 2))))

        # Convert predicted log h -> policy -> terminal distribution.
        h_true = np.exp(np.clip(y_true, -500, 500))
        h_hat = np.exp(np.clip(y_hat, -500, 500))
        pi_true = dag.induced_policy_from_h(h_true, pi0)
        pi_hat = dag.induced_policy_from_h(h_hat, pi0)
        td_true = dag.terminal_distribution_from_policy(pi_true)
        td_hat = dag.terminal_distribution_from_policy(pi_hat)
        kl_terms.append(kl_divergence(td_true, td_hat))

    return {
        "setup": {
            "depth": depth,
            "n_internal": dag.n_internal,
            "n_terminals": dag.n_terminals,
            "feature_dim": feature_dim,
            "seed": seed,
        },
        "q1_linear": {
            "max_abs_error": linear_max_abs_error,
            "mean_abs_error": linear_mean_abs_error,
        },
        "q1_loglinear_global_fit": {
            "state_rmse_log_mean": float(np.mean(rmse_log)),
            "state_rmse_log_max": float(np.max(rmse_log)),
            "terminal_kl_mean": float(np.mean(kl_terms)),
            "terminal_kl_max": float(np.max(kl_terms)),
        },
    }


def main() -> None:
    out_dir = ROOT / "outputs" / "binary_dag"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_q1()
    out_path = out_dir / "q1_binary_dag.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
