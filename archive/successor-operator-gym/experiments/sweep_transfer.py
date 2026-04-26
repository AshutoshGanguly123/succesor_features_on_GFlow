#!/usr/bin/env python3
"""Transfer sweep: compare all methods across many random reward perturbations.

Generates aggregate plots: KL vs delta_norm, runtime comparisons.
"""

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
from successor_operator.projection import projected_policy, approximate_h_from_geometry
from successor_operator.rollout import compute_terminal_distribution_model
from successor_operator.metrics import terminal_kl
from successor_operator.utils import load_config, save_results, timed, console, parallel_map


def sweep_single(args):
    """Evaluate a single random delta."""
    (mdp, pi0, step_feat, term_feat, w0, delta, ref_h, ref_grad, ref_hess, A0) = args

    w_new = w0 + delta
    delta_norm = float(np.linalg.norm(delta))

    try:
        h_exact, t_exact = timed(solve_successor_mass, mdp, pi0, step_feat, term_feat, w_new)
        pi_exact = projected_policy(mdp, pi0, step_feat, w_new, h_exact)
        td_exact = compute_terminal_distribution_model(mdp, pi_exact)

        h_o1 = approximate_h_from_geometry(A0, ref_grad, ref_hess, delta, order=1)
        pi_o1 = projected_policy(mdp, pi0, step_feat, w_new, h_o1)
        td_o1 = compute_terminal_distribution_model(mdp, pi_o1)

        h_o2 = approximate_h_from_geometry(A0, ref_grad, ref_hess, delta, order=2)
        pi_o2 = projected_policy(mdp, pi0, step_feat, w_new, h_o2)
        td_o2 = compute_terminal_distribution_model(mdp, pi_o2)

        pi_naive = projected_policy(mdp, pi0, step_feat, w0, ref_h)
        td_naive = compute_terminal_distribution_model(mdp, pi_naive)

        return {
            "delta_norm": delta_norm,
            "exact_time": t_exact,
            "kl_order1": terminal_kl(td_exact, td_o1),
            "kl_order2": terminal_kl(td_exact, td_o2),
            "kl_naive": terminal_kl(td_exact, td_naive),
        }
    except Exception:
        return {
            "delta_norm": delta_norm,
            "exact_time": float("nan"),
            "kl_order1": float("nan"),
            "kl_order2": float("nan"),
            "kl_naive": float("nan"),
        }


def run_transfer_sweep(env_name, env_kwargs, feature_fn, feature_args,
                       base_w, n_deltas, max_norm, seed, n_workers):
    """Full transfer sweep for one environment."""
    mdp = extract_tabular_mdp(env_name, **env_kwargs)
    mdp.validate()

    if feature_args:
        step_feat, term_feat = feature_fn(mdp, *feature_args)
    else:
        step_feat, term_feat = feature_fn(mdp)

    pi0 = uniform_passive_policy(mdp)
    w0 = np.array(base_w, dtype=np.float64)
    d = len(w0)

    console.print(f"  Solving reference (w={w0.tolist()})...")
    ref_h = solve_successor_mass(mdp, pi0, step_feat, term_feat, w0)
    ref_grad, ref_hess = solve_moments_vectorized(mdp, pi0, step_feat, term_feat, w0, ref_h)
    A0 = np.log(np.maximum(ref_h, 1e-300))

    rng = np.random.default_rng(seed)
    norms = np.linspace(0.1, max_norm, n_deltas)
    deltas = []
    for norm_target in norms:
        direction = rng.standard_normal(d)
        direction = direction / np.linalg.norm(direction) * norm_target
        deltas.append(direction)

    args_list = [
        (mdp, pi0, step_feat, term_feat, w0, delta, ref_h, ref_grad, ref_hess, A0)
        for delta in deltas
    ]

    console.print(f"  Sweeping {n_deltas} deltas with {n_workers} workers...")
    results = parallel_map(sweep_single, args_list, n_workers=n_workers)
    return results


def plot_transfer(results, title, output_path):
    """Plot terminal KL vs delta norm for all methods."""
    valid = [r for r in results if np.isfinite(r["kl_order1"]) and np.isfinite(r["kl_order2"])]
    if not valid:
        console.print(f"  [yellow]No valid results to plot for {title}")
        return
    norms = [r["delta_norm"] for r in valid]
    kl_o1 = [r["kl_order1"] for r in valid]
    kl_o2 = [r["kl_order2"] for r in valid]
    kl_naive = [r["kl_naive"] for r in valid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(norms, kl_o1, "o-", label="1st-order geometry", color="#e74c3c", markersize=4)
    ax.plot(norms, kl_o2, "s-", label="2nd-order geometry", color="#2ecc71", markersize=4)
    ax.plot(norms, kl_naive, "^--", label="Naive reuse", color="#95a5a6", markersize=4)
    ax.set_xlabel("||δ|| (reward shift norm)")
    ax.set_ylabel("Terminal KL divergence")
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
    n_workers = 6

    # FrozenLake
    fl_config = load_config(os.path.join(configs_dir, "frozenlake.yaml"))
    geo = fl_config["geometry_sweep"]
    for variant in fl_config["variants"]:
        console.rule(f"[bold]Transfer sweep: FrozenLake {variant['name']}")
        env_kwargs = variant.get("env_kwargs", {})
        desc = get_env_desc("FrozenLake-v1", **env_kwargs)
        results = run_transfer_sweep(
            "FrozenLake-v1", env_kwargs,
            frozenlake_features, (desc,),
            geo["base_w"], geo["n_deltas"],
            abs(geo["delta_range"][1] - geo["delta_range"][0]),
            geo["seed"], n_workers
        )
        plot_transfer(results, f"FrozenLake {variant['name']}: Terminal KL vs Reward Shift",
                     os.path.join(figs_dir, f"transfer_frozenlake_{variant['name']}.png"))

    # CliffWalking
    cw_config = load_config(os.path.join(configs_dir, "cliffwalking.yaml"))
    geo = cw_config["geometry_sweep"]
    for variant in cw_config["variants"]:
        console.rule(f"[bold]Transfer sweep: CliffWalking {variant['name']}")
        results = run_transfer_sweep(
            cw_config["env_name"], variant.get("env_kwargs", {}),
            cliffwalking_features, (),
            geo["base_w"], geo["n_deltas"],
            abs(geo["delta_range"][1] - geo["delta_range"][0]),
            geo["seed"], n_workers
        )
        plot_transfer(results, f"CliffWalking {variant['name']}: Terminal KL vs Reward Shift",
                     os.path.join(figs_dir, f"transfer_cliffwalking_{variant['name']}.png"))

    # Taxi
    tx_config = load_config(os.path.join(configs_dir, "taxi.yaml"))
    geo = tx_config["geometry_sweep"]
    for variant in tx_config["variants"]:
        console.rule(f"[bold]Transfer sweep: Taxi {variant['name']}")
        results = run_transfer_sweep(
            tx_config["env_name"], variant.get("env_kwargs", {}),
            taxi_features, (),
            geo["base_w"], geo["n_deltas"],
            abs(geo["delta_range"][1] - geo["delta_range"][0]),
            geo["seed"], n_workers
        )
        plot_transfer(results, f"Taxi {variant['name']}: Terminal KL vs Reward Shift",
                     os.path.join(figs_dir, f"transfer_taxi_{variant['name']}.png"))

    console.print("\n[green bold]All transfer sweeps complete!")


if __name__ == "__main__":
    main()
