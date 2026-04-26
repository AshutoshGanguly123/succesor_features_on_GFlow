#!/usr/bin/env python3
"""Geometry sweep: measure first-order vs second-order approximation error
across a range of reward perturbation magnitudes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from successor_operator.gym_extract import extract_tabular_mdp, get_env_desc
from successor_operator.passive_policy import uniform_passive_policy
from successor_operator.features import frozenlake_features, cliffwalking_features, taxi_features
from successor_operator.feynman_kac import solve_successor_mass
from successor_operator.moments import solve_moments_vectorized
from successor_operator.projection import approximate_h_from_geometry
from successor_operator.metrics import geometry_error
from successor_operator.utils import load_config, save_results, timed, console, parallel_map


def sweep_for_env(env_name, env_kwargs, feature_fn, feature_args, base_w, n_deltas, delta_range, seed):
    """Run a geometry sweep for one environment variant."""
    mdp = extract_tabular_mdp(env_name, **env_kwargs)
    mdp.validate()

    step_feat, term_feat = feature_fn(mdp, *feature_args) if feature_args else feature_fn(mdp)
    pi0 = uniform_passive_policy(mdp)
    w0 = np.array(base_w, dtype=np.float64)
    d = len(w0)

    h0 = solve_successor_mass(mdp, pi0, step_feat, term_feat, w0)
    grad0, hess0 = solve_moments_vectorized(mdp, pi0, step_feat, term_feat, w0, h0)
    A0 = np.log(np.maximum(h0, 1e-300))

    rng = np.random.default_rng(seed)
    norms = np.linspace(0.01, abs(delta_range[1] - delta_range[0]), n_deltas)

    results = []
    for norm_target in norms:
        direction = rng.standard_normal(d)
        direction = direction / np.linalg.norm(direction) * norm_target
        delta = direction

        w_new = w0 + delta
        h_exact = solve_successor_mass(mdp, pi0, step_feat, term_feat, w_new)

        h_o1 = approximate_h_from_geometry(A0, grad0, hess0, delta, order=1)
        h_o2 = approximate_h_from_geometry(A0, grad0, hess0, delta, order=2)

        err_o1 = geometry_error(h_exact, h_o1)
        err_o2 = geometry_error(h_exact, h_o2)

        results.append({
            "delta_norm": float(norm_target),
            "rmse_order1": err_o1["rmse_log"],
            "rmse_order2": err_o2["rmse_log"],
            "linf_order1": err_o1["linf_log"],
            "linf_order2": err_o2["linf_log"],
        })

    return results


def plot_sweep(results, title, output_path):
    """Generate geometry error vs delta norm plot."""
    norms = [r["delta_norm"] for r in results]
    rmse_o1 = [r["rmse_order1"] for r in results]
    rmse_o2 = [r["rmse_order2"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(norms, rmse_o1, "o-", label="First-order", color="#e74c3c", markersize=4)
    ax.plot(norms, rmse_o2, "s-", label="Second-order", color="#2ecc71", markersize=4)
    ax.set_xlabel("||δ|| (reward shift norm)")
    ax.set_ylabel("RMSE(log h)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Plot saved to {output_path}")


def main():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    figs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

    # FrozenLake sweeps
    fl_config = load_config(os.path.join(configs_dir, "frozenlake.yaml"))
    geo = fl_config["geometry_sweep"]
    for variant in fl_config["variants"]:
        console.rule(f"[bold]Geometry sweep: FrozenLake {variant['name']}")
        env_kwargs = variant.get("env_kwargs", {})
        desc = get_env_desc("FrozenLake-v1", **env_kwargs)

        results = sweep_for_env(
            "FrozenLake-v1", env_kwargs,
            frozenlake_features, (desc,),
            geo["base_w"], geo["n_deltas"], geo["delta_range"], geo["seed"]
        )
        plot_sweep(results, f"FrozenLake {variant['name']}: Geometry Approximation",
                  os.path.join(figs_dir, f"geometry_frozenlake_{variant['name']}.png"))

    # CliffWalking sweep
    cw_config = load_config(os.path.join(configs_dir, "cliffwalking.yaml"))
    geo = cw_config["geometry_sweep"]
    for variant in cw_config["variants"]:
        console.rule(f"[bold]Geometry sweep: CliffWalking {variant['name']}")
        results = sweep_for_env(
            cw_config["env_name"], variant.get("env_kwargs", {}),
            cliffwalking_features, (),
            geo["base_w"], geo["n_deltas"], geo["delta_range"], geo["seed"]
        )
        plot_sweep(results, f"CliffWalking {variant['name']}: Geometry Approximation",
                  os.path.join(figs_dir, f"geometry_cliffwalking_{variant['name']}.png"))

    # Taxi sweep
    tx_config = load_config(os.path.join(configs_dir, "taxi.yaml"))
    geo = tx_config["geometry_sweep"]
    for variant in tx_config["variants"]:
        console.rule(f"[bold]Geometry sweep: Taxi {variant['name']}")
        results = sweep_for_env(
            tx_config["env_name"], variant.get("env_kwargs", {}),
            taxi_features, (),
            geo["base_w"], geo["n_deltas"], geo["delta_range"], geo["seed"]
        )
        plot_sweep(results, f"Taxi {variant['name']}: Geometry Approximation",
                  os.path.join(figs_dir, f"geometry_taxi_{variant['name']}.png"))

    console.print("\n[green bold]All geometry sweeps complete!")


if __name__ == "__main__":
    main()
