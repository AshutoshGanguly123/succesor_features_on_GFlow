#!/usr/bin/env python3
"""Stage E: Taxi horizon sweep. Run H ∈ {20, 40, 80} and compare stability."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

from src.envs.build_tabular_model import extract_tabular_model, uniform_policy
from src.envs.taxi_features import build_taxi_reward
from src.core.dp import solve_h
from src.core.bridge_projection import project_policy
from src.core.geometry import solve_moments
from src.core.occupancy import (
    compute_terminal_distribution, compute_occupancy, compute_mean_episode_length
)
from src.core.metrics import terminal_kl, geometry_error
from src.core.atlas import build_anchor
from src.core.geometry import approx_h
from src.utils.io import load_config, save_results
from src.utils.logging import timed, print_results_table, console


def run_horizon(H: int, tasks: list, anchor_r: np.ndarray) -> dict:
    console.rule(f"[bold]Taxi H={H}")

    model = extract_tabular_model("Taxi-v3", H)
    model.validate()
    reward = build_taxi_reward(model)
    pi0 = uniform_policy(model)

    console.print(f"  Building anchor (H={H})...")
    anchor, t_build = timed(build_anchor, "standard", anchor_r, model, reward, pi0)
    console.print(f"  Anchor: {t_build:.3f}s, h range [{anchor.h.min():.4g}, {anchor.h.max():.4g}]")

    h_finite = np.all(np.isfinite(anchor.h))
    mu_finite = np.all(np.isfinite(anchor.mu))
    console.print(f"  h finite: {h_finite}, mu finite: {mu_finite}")

    results = []
    for task in tasks:
        r_t = np.array(task["r"], dtype=np.float64)
        res = {"task": task["name"], "horizon": H}

        h_exact, t_solve = timed(solve_h, model, reward, pi0, r_t)
        res["exact_solve_time"] = t_solve
        res["h_finite"] = bool(np.all(np.isfinite(h_exact)))
        res["h_max"] = float(np.max(h_exact[np.isfinite(h_exact)])) if res["h_finite"] else float("inf")

        pi_exact, _ = project_policy(model, reward, pi0, r_t, h_exact)
        td_exact = compute_terminal_distribution(model, pi_exact)
        res["mean_episode_length"] = compute_mean_episode_length(model, pi_exact)

        # Second-order transfer from anchor
        h_approx = approx_h(anchor, r_t, order=2)
        pi_approx, _ = project_policy(model, reward, pi0, r_t, h_approx)
        td_approx = compute_terminal_distribution(model, pi_approx)

        res["terminal_kl_o2"] = terminal_kl(td_exact, td_approx)
        res["geometry_error"] = geometry_error(h_exact, h_approx)
        res["delta_norm"] = float(np.linalg.norm(r_t - anchor_r))

        results.append(res)

    summary = [{
        "task": r["task"], "H": r["horizon"],
        "t_solve": r["exact_solve_time"],
        "h_ok": r["h_finite"],
        "h_max": r["h_max"],
        "ep_len": r["mean_episode_length"],
        "KL_t_o2": r["terminal_kl_o2"],
    } for r in results]
    print_results_table(summary, title=f"Taxi H={H}")

    return {"horizon": H, "anchor_build_time": t_build,
            "h_finite": h_finite, "mu_finite": mu_finite,
            "tasks": results}


def main():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

    cfg = load_config(os.path.join(configs_dir, "taxi_h20.yaml"))
    anchor_r = np.array(cfg["anchors"][0]["r"], dtype=np.float64)
    tasks = cfg["tasks"]

    all_horizon_results = []
    for H in [20, 40, 80]:
        result = run_horizon(H, tasks, anchor_r)
        all_horizon_results.append(result)

    save_results(
        {"horizons": all_horizon_results},
        os.path.join(outputs_dir, "taxi_horizon_sweep.json")
    )
    console.print("\n[green bold]Stage E (Taxi Horizon Sweep) complete!")


if __name__ == "__main__":
    main()
