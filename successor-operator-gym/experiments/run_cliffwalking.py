#!/usr/bin/env python3
"""CliffWalking experiment: exact solve, projected policy, geometry transfer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

from successor_operator.gym_extract import extract_tabular_mdp
from successor_operator.passive_policy import uniform_passive_policy
from successor_operator.features import cliffwalking_features
from successor_operator.feynman_kac import solve_successor_mass
from successor_operator.moments import solve_moments_vectorized
from successor_operator.projection import (
    projected_policy, gold_standard_tilted_law, approximate_h_from_geometry
)
from successor_operator.rollout import compute_terminal_distribution_model, compute_state_occupancy
from successor_operator.metrics import (
    terminal_kl, occupancy_kl, realizability_gap, geometry_error
)
from successor_operator.utils import (
    load_config, save_results, timed, print_results_table, parallel_map, console
)


def run_single_task(args):
    """Run a single task evaluation."""
    mdp, pi0, step_feat, term_feat, w, task_name, ref_w, ref_h, ref_grad, ref_hess = args

    results = {"task": task_name, "w": w.tolist()}

    h_exact, t_exact = timed(solve_successor_mass, mdp, pi0, step_feat, term_feat, w)
    results["exact_solve_time"] = t_exact

    pi_exact = projected_policy(mdp, pi0, step_feat, w, h_exact)
    q_tilted = gold_standard_tilted_law(mdp, pi0, step_feat, w, h_exact)

    term_dist_exact = compute_terminal_distribution_model(mdp, pi_exact)
    occ_exact = compute_state_occupancy(mdp, pi_exact)

    delta = w - ref_w
    A_ref = np.log(np.maximum(ref_h, 1e-300))

    h_order1 = approximate_h_from_geometry(A_ref, ref_grad, ref_hess, delta, order=1)
    pi_order1 = projected_policy(mdp, pi0, step_feat, w, h_order1)
    term_dist_o1 = compute_terminal_distribution_model(mdp, pi_order1)
    occ_o1 = compute_state_occupancy(mdp, pi_order1)

    h_order2 = approximate_h_from_geometry(A_ref, ref_grad, ref_hess, delta, order=2)
    pi_order2 = projected_policy(mdp, pi0, step_feat, w, h_order2)
    term_dist_o2 = compute_terminal_distribution_model(mdp, pi_order2)
    occ_o2 = compute_state_occupancy(mdp, pi_order2)

    pi_ref = projected_policy(mdp, pi0, step_feat, ref_w, ref_h)
    term_dist_naive = compute_terminal_distribution_model(mdp, pi_ref)

    results["terminal_kl_order1"] = terminal_kl(term_dist_exact, term_dist_o1)
    results["terminal_kl_order2"] = terminal_kl(term_dist_exact, term_dist_o2)
    results["terminal_kl_naive"] = terminal_kl(term_dist_exact, term_dist_naive)
    results["occupancy_kl_order1"] = occupancy_kl(occ_exact, occ_o1)
    results["occupancy_kl_order2"] = occupancy_kl(occ_exact, occ_o2)
    results["geometry_error_order1"] = geometry_error(h_exact, h_order1)
    results["geometry_error_order2"] = geometry_error(h_exact, h_order2)

    gap = realizability_gap(mdp, q_tilted, pi_exact)
    nt = ~mdp.terminal
    results["realizability_gap_mean"] = float(gap[nt].mean()) if nt.any() else 0.0
    results["realizability_gap_max"] = float(gap[nt].max()) if nt.any() else 0.0
    results["delta_norm"] = float(np.linalg.norm(delta))

    return results


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "cliffwalking.yaml")
    config = load_config(config_path)

    env_name = config["env_name"]
    n_workers = config.get("n_workers", 6)

    all_variant_results = []
    for variant in config["variants"]:
        env_kwargs = variant.get("env_kwargs", {})
        variant_name = variant["name"]
        console.rule(f"[bold blue]{env_name} - {variant_name}")

        mdp = extract_tabular_mdp(env_name, **env_kwargs)
        mdp.validate()
        console.print(f"  States: {mdp.nS}, Actions: {mdp.nA}, "
                      f"Terminal: {mdp.terminal.sum()}, Nonterminal: {(~mdp.terminal).sum()}")

        step_feat, term_feat = cliffwalking_features(mdp)
        pi0 = uniform_passive_policy(mdp)
        ref_w = np.array(config["reference_w"], dtype=np.float64)

        console.print(f"  Solving reference task w={ref_w.tolist()}...")
        ref_h, t_ref = timed(solve_successor_mass, mdp, pi0, step_feat, term_feat, ref_w)
        console.print(f"  Reference solve: {t_ref:.4f}s")

        (ref_grad, ref_hess), t_mom = timed(
            solve_moments_vectorized, mdp, pi0, step_feat, term_feat, ref_w, ref_h
        )
        console.print(f"  Moments solve: {t_mom:.4f}s")

        task_args = []
        for tw in config["task_weights"]:
            w = np.array(tw["w"], dtype=np.float64)
            task_args.append((mdp, pi0, step_feat, term_feat, w, tw["name"],
                             ref_w, ref_h, ref_grad, ref_hess))

        console.print(f"  Running {len(task_args)} tasks with {n_workers} workers...")
        all_results = parallel_map(run_single_task, task_args, n_workers=n_workers)

        summary = []
        for r in all_results:
            summary.append({
                "task": r["task"],
                "|δ|": r["delta_norm"],
                "t_solve": r["exact_solve_time"],
                "KL_term_o1": r["terminal_kl_order1"],
                "KL_term_o2": r["terminal_kl_order2"],
                "KL_term_naive": r["terminal_kl_naive"],
                "gap_mean": r["realizability_gap_mean"],
                "gap_max": r["realizability_gap_max"],
            })
        print_results_table(summary, title=f"{env_name} / {variant_name}")

        all_variant_results.append({
            "variant": variant_name,
            "nS": mdp.nS, "nA": mdp.nA,
            "reference_solve_time": t_ref,
            "moments_solve_time": t_mom,
            "tasks": all_results,
        })

    output_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "cliffwalking_results.json")
    save_results({"env": env_name, "variants": all_variant_results}, output_path)
    console.print(f"\n[green]Results saved to {output_path}")


if __name__ == "__main__":
    main()
