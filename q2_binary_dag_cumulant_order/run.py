#!/usr/bin/env python3
"""Q2: first-order vs second-order cumulant approximation on binary DAG."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "lib"))

from src.envs.binary_dag import BinaryDAG, kl_divergence


def tilted_moments(
    dag: BinaryDAG, mu: np.ndarray, terminal_rewards_w0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h0 = mu @ terminal_rewards_w0
    wts = (mu * terminal_rewards_w0[None, :]) / np.maximum(h0[:, None], 1e-300)
    mu_t = wts @ dag.terminal_features
    second = np.einsum("sl,ld,le->sde", wts, dag.terminal_features, dag.terminal_features)
    cov = second - np.einsum("sd,se->sde", mu_t, mu_t)
    return mu_t, cov


def run_q2(
    depth: int = 8,
    feature_dim: int = 6,
    seed: int = 9,
    n_deltas: int = 12,
    delta_max: float = 1.2,
) -> dict:
    dag = BinaryDAG.from_random(depth=depth, feature_dim=feature_dim, seed=seed)
    rng = np.random.default_rng(seed + 101)
    w0 = rng.normal(size=feature_dim)
    _, pi0 = dag.reference_flow_and_policy(w0)
    mu = dag.descendant_measure(pi0)

    r0 = dag.leaf_rewards(w0, "loglinear")
    log_h0 = dag.log_h_from_mu(mu, w0, "loglinear")
    mean_tilt, cov_tilt = tilted_moments(dag, mu, r0)

    direction = rng.normal(size=feature_dim)
    direction /= max(np.linalg.norm(direction), 1e-12)
    delta_norms = np.linspace(0.1, delta_max, n_deltas)

    rows = []
    for dn in delta_norms:
        delta = dn * direction
        w = w0 + delta
        true_log_h = dag.log_h_from_mu(mu, w, "loglinear")
        pred_log_o1 = log_h0 + mean_tilt @ delta
        quad = 0.5 * np.einsum("i,sij,j->s", delta, cov_tilt, delta)
        pred_log_o2 = pred_log_o1 + quad

        h_true = np.exp(np.clip(true_log_h, -500, 500))
        h_o1 = np.exp(np.clip(pred_log_o1, -500, 500))
        h_o2 = np.exp(np.clip(pred_log_o2, -500, 500))

        pi_true = dag.induced_policy_from_h(h_true, pi0)
        pi_o1 = dag.induced_policy_from_h(h_o1, pi0)
        pi_o2 = dag.induced_policy_from_h(h_o2, pi0)

        td_true = dag.terminal_distribution_from_policy(pi_true)
        td_o1 = dag.terminal_distribution_from_policy(pi_o1)
        td_o2 = dag.terminal_distribution_from_policy(pi_o2)

        kl1 = kl_divergence(td_true, td_o1)
        kl2 = kl_divergence(td_true, td_o2)
        rows.append(
            {
                "delta_norm": float(dn),
                "terminal_kl_o1": float(kl1),
                "terminal_kl_o2": float(kl2),
                "improvement_ratio_o1_over_o2": float(kl1 / max(kl2, 1e-300)),
            }
        )

    return {
        "setup": {
            "depth": depth,
            "n_internal": dag.n_internal,
            "n_terminals": dag.n_terminals,
            "feature_dim": feature_dim,
            "seed": seed,
        },
        "q2_results": rows,
    }


def main() -> None:
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_q2()
    out_path = out_dir / "q2_binary_dag.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
