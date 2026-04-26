#!/usr/bin/env python3
"""Q3: warm-start comparison at zero optimization steps on binary DAG."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reward_atlas_transfer"))

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


def run_q3(
    depth: int = 8,
    feature_dim: int = 6,
    seed: int = 17,
    n_targets: int = 6,
    delta_scale: float = 0.8,
) -> dict:
    dag = BinaryDAG.from_random(depth=depth, feature_dim=feature_dim, seed=seed)
    rng = np.random.default_rng(seed + 123)
    w0 = rng.normal(size=feature_dim)
    _, pi0 = dag.reference_flow_and_policy(w0)
    mu = dag.descendant_measure(pi0)

    r0 = dag.leaf_rewards(w0, "loglinear")
    log_h0 = dag.log_h_from_mu(mu, w0, "loglinear")
    mean_tilt, cov_tilt = tilted_moments(dag, mu, r0)

    rows = []
    for i in range(n_targets):
        delta = rng.normal(size=feature_dim)
        delta = delta / max(np.linalg.norm(delta), 1e-12) * delta_scale
        w = w0 + delta

        true_log_h = dag.log_h_from_mu(mu, w, "loglinear")
        h_true = np.exp(np.clip(true_log_h, -500, 500))
        pi_true = dag.induced_policy_from_h(h_true, pi0)
        td_true = dag.terminal_distribution_from_policy(pi_true)

        # Scratch (uniform)
        pi_scratch = np.full((dag.n_internal, 2), 0.5, dtype=np.float64)
        td_scratch = dag.terminal_distribution_from_policy(pi_scratch)

        # Fine-tune start: reference policy
        td_finetune = dag.terminal_distribution_from_policy(pi0)

        # First-order warm-start from cumulant expansion
        log_h_o1 = log_h0 + mean_tilt @ delta
        h_o1 = np.exp(np.clip(log_h_o1, -500, 500))
        pi_o1 = dag.induced_policy_from_h(h_o1, pi0)
        td_o1 = dag.terminal_distribution_from_policy(pi_o1)

        # Second-order warm-start
        quad = 0.5 * np.einsum("i,sij,j->s", delta, cov_tilt, delta)
        h_o2 = np.exp(np.clip(log_h_o1 + quad, -500, 500))
        pi_o2 = dag.induced_policy_from_h(h_o2, pi0)
        td_o2 = dag.terminal_distribution_from_policy(pi_o2)

        rows.append(
            {
                "task_id": i,
                "delta_norm": float(np.linalg.norm(delta)),
                "kl_scratch": kl_divergence(td_true, td_scratch),
                "kl_finetune": kl_divergence(td_true, td_finetune),
                "kl_first_order": kl_divergence(td_true, td_o1),
                "kl_second_order": kl_divergence(td_true, td_o2),
            }
        )

    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "setup": {
            "depth": depth,
            "n_internal": dag.n_internal,
            "n_terminals": dag.n_terminals,
            "feature_dim": feature_dim,
            "seed": seed,
        },
        "q3_zero_step_table": rows,
        "q3_zero_step_summary_mean_kl": {
            "scratch": avg("kl_scratch"),
            "fine_tune": avg("kl_finetune"),
            "first_order": avg("kl_first_order"),
            "second_order": avg("kl_second_order"),
        },
    }


def main() -> None:
    out_dir = ROOT / "outputs" / "binary_dag"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_q3()
    out_path = out_dir / "q3_binary_dag.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
