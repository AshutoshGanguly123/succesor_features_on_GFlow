#!/usr/bin/env python3
"""Stage A: Direct transfer experiments.

For each environment config, runs:
- exact oracle solve for each task
- naive reuse from first anchor
- single-anchor first/second order
- reports terminal KL, occupancy KL, geometry errors, realizability gap
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from pathlib import Path

from src.envs.build_tabular_model import extract_tabular_model, uniform_policy
from src.envs.frozenlake_features import build_frozenlake_reward
from src.envs.cliffwalking_features import build_cliffwalking_reward
from src.envs.taxi_features import build_taxi_reward
from src.core.dp import solve_h
from src.core.bridge_projection import project_policy, tilted_joint_law
from src.core.geometry import solve_moments, approx_h
from src.core.occupancy import compute_terminal_distribution, compute_occupancy
from src.core.metrics import terminal_kl, occupancy_kl, geometry_error, policy_kl
from src.core.diagnostics import (
    realizability_gap, first_second_disagreement, min_action_margin,
    action_flip_rate, curvature_score
)
from src.core.atlas import build_anchor
from src.utils.io import load_config, save_results
from src.utils.logging import timed, print_results_table, console


ENV_FEATURE_BUILDERS = {
    "FrozenLake-v1": lambda model, cfg: build_frozenlake_reward(model, cfg.get("env_kwargs", {})),
    "CliffWalking-v1": lambda model, cfg: build_cliffwalking_reward(model),
    "Taxi-v3": lambda model, cfg: build_taxi_reward(model),
}


def run_task(model, reward, pi0, r_target, task_name, anchor):
    """Evaluate a single target task against an anchor."""
    result = {"task": task_name, "r": r_target.tolist()}

    # Oracle: exact solve
    h_exact, t_solve = timed(solve_h, model, reward, pi0, r_target)
    pi_exact, _ = project_policy(model, reward, pi0, r_target, h_exact)
    result["exact_solve_time"] = t_solve

    td_exact = compute_terminal_distribution(model, pi_exact)
    occ_exact = compute_occupancy(model, pi_exact)

    # Naive reuse
    td_naive = compute_terminal_distribution(model, anchor.policy)
    occ_naive = compute_occupancy(model, anchor.policy)
    result["terminal_kl_naive"] = terminal_kl(td_exact, td_naive)
    result["occupancy_kl_naive"] = occupancy_kl(occ_exact, occ_naive)

    # First-order from anchor
    h_o1 = approx_h(anchor, r_target, order=1)
    pi_o1, _ = project_policy(model, reward, pi0, r_target, h_o1)
    td_o1 = compute_terminal_distribution(model, pi_o1)
    occ_o1 = compute_occupancy(model, pi_o1)
    result["terminal_kl_single_o1"] = terminal_kl(td_exact, td_o1)
    result["occupancy_kl_single_o1"] = occupancy_kl(occ_exact, occ_o1)
    result["geometry_error_o1"] = geometry_error(h_exact, h_o1)

    # Second-order from anchor
    h_o2 = approx_h(anchor, r_target, order=2)
    pi_o2, _ = project_policy(model, reward, pi0, r_target, h_o2)
    td_o2 = compute_terminal_distribution(model, pi_o2)
    occ_o2 = compute_occupancy(model, pi_o2)
    result["terminal_kl_single_o2"] = terminal_kl(td_exact, td_o2)
    result["occupancy_kl_single_o2"] = occupancy_kl(occ_exact, occ_o2)
    result["geometry_error_o2"] = geometry_error(h_exact, h_o2)

    # Policy KL
    result["policy_kl_o1"] = policy_kl(pi_exact, pi_o1, model)
    result["policy_kl_o2"] = policy_kl(pi_exact, pi_o2, model)

    # Diagnostics
    q_tilted = tilted_joint_law(model, reward, pi0, r_target, h_exact)
    gap = realizability_gap(model, q_tilted, pi_exact)
    nt = model.nonterminal_mask()
    result["realizability_gap_mean"] = float(gap[:, nt].mean()) if nt.any() else 0.0
    result["realizability_gap_max"] = float(gap[:, nt].max()) if nt.any() else 0.0
    result["curvature_score"] = curvature_score(anchor, r_target)
    result["first_second_disagreement"] = first_second_disagreement(anchor, r_target, model)
    result["min_action_margin"] = min_action_margin(model, pi_exact)
    result["action_flip_rate"] = action_flip_rate(model, anchor.policy, pi_exact)
    result["delta_norm"] = float(np.linalg.norm(r_target - anchor.r))

    return result


def run_config(config_path: str) -> dict:
    """Run Stage A for a single config file."""
    cfg = load_config(config_path)
    env_name = cfg["env_name"]
    H = cfg["horizon"]
    variant = cfg["variant"]

    console.rule(f"[bold blue]{env_name} / {variant} (H={H})")

    model = extract_tabular_model(env_name, H, **cfg.get("env_kwargs", {}))
    model.validate()
    console.print(f"  States: {model.nS}, Actions: {model.nA}, H: {H}, "
                  f"Terminal: {model.terminal.sum()}")

    builder = ENV_FEATURE_BUILDERS[env_name]
    reward = builder(model, cfg)
    pi0 = uniform_policy(model)

    # Build primary anchor (first in list)
    anchor_spec = cfg["anchors"][0]
    r_anchor = np.array(anchor_spec["r"], dtype=np.float64)
    console.print(f"  Building anchor '{anchor_spec['name']}' at r={r_anchor.tolist()}...")
    anchor, t_anchor = timed(build_anchor, anchor_spec["name"], r_anchor,
                              model, reward, pi0)
    console.print(f"  Anchor build: {t_anchor:.3f}s")

    # Run all tasks
    all_results = []
    for task in cfg["tasks"]:
        r_t = np.array(task["r"], dtype=np.float64)
        res = run_task(model, reward, pi0, r_t, task["name"], anchor)
        all_results.append(res)

    # Summary table
    summary = []
    for r in all_results:
        summary.append({
            "task": r["task"],
            "|δ|": r["delta_norm"],
            "t_solve": r["exact_solve_time"],
            "KL_t_naive": r["terminal_kl_naive"],
            "KL_t_o1": r["terminal_kl_single_o1"],
            "KL_t_o2": r["terminal_kl_single_o2"],
            "KL_occ_o2": r["occupancy_kl_single_o2"],
            "gap_mean": r["realizability_gap_mean"],
        })
    print_results_table(summary, title=f"{env_name} / {variant} - Direct Transfer")

    return {
        "env": env_name, "variant": variant, "horizon": H,
        "anchor": anchor_spec["name"],
        "anchor_build_time": t_anchor,
        "nS": model.nS, "nA": model.nA,
        "tasks": all_results,
    }


def main():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

    config_files = sorted(Path(configs_dir).glob("*.yaml"))
    all_env_results = []

    for cfg_path in config_files:
        try:
            result = run_config(str(cfg_path))
            all_env_results.append(result)
            save_results(
                result,
                os.path.join(outputs_dir, f"direct_{result['variant']}.json")
            )
        except Exception as e:
            console.print(f"[red]  Error on {cfg_path.name}: {e}")

    console.print("\n[green bold]Stage A complete!")


if __name__ == "__main__":
    main()
