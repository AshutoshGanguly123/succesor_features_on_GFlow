#!/usr/bin/env python3
"""Stage B: Atlas-direct transfer experiments.

For each target reward, select the best anchor by geometric distance,
then run first/second order local transfer from that anchor.
Compare against single-anchor methods.
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
from src.core.bridge_projection import project_policy
from src.core.geometry import approx_h
from src.core.atlas import build_atlas, select_anchor, transfer_from_anchor, anchor_distance
from src.core.occupancy import compute_terminal_distribution, compute_occupancy
from src.core.metrics import terminal_kl, occupancy_kl, geometry_error
from src.core.diagnostics import compute_all_diagnostics
from src.utils.io import load_config, save_results
from src.utils.logging import timed, print_results_table, console


ENV_FEATURE_BUILDERS = {
    "FrozenLake-v1": lambda model, cfg: build_frozenlake_reward(model, cfg.get("env_kwargs", {})),
    "CliffWalking-v1": lambda model, cfg: build_cliffwalking_reward(model),
    "Taxi-v3": lambda model, cfg: build_taxi_reward(model),
}


def run_config(config_path: str) -> dict:
    cfg = load_config(config_path)
    env_name = cfg["env_name"]
    H = cfg["horizon"]
    variant = cfg["variant"]

    console.rule(f"[bold blue]{env_name} / {variant} - Atlas Transfer")

    model = extract_tabular_model(env_name, H, **cfg.get("env_kwargs", {}))
    model.validate()
    builder = ENV_FEATURE_BUILDERS[env_name]
    reward = builder(model, cfg)
    pi0 = uniform_policy(model)

    console.print(f"  Building atlas with {len(cfg['anchors'])} anchors...")
    trust_r = cfg.get("trust_radius", 2.0)
    atlas, t_atlas = timed(build_atlas, cfg["anchors"], model, reward, pi0, trust_r)
    console.print(f"  Atlas built in {t_atlas:.3f}s")

    # Single reference anchor (first)
    single_anchor = atlas[0]

    all_results = []
    for task in cfg["tasks"]:
        r_t = np.array(task["r"], dtype=np.float64)
        res = {"task": task["name"], "r": r_t.tolist()}

        # Oracle
        h_exact = solve_h(model, reward, pi0, r_t)
        pi_exact, _ = project_policy(model, reward, pi0, r_t, h_exact)
        td_exact = compute_terminal_distribution(model, pi_exact)
        occ_exact = compute_occupancy(model, pi_exact)

        # Single-anchor second-order
        h_single = approx_h(single_anchor, r_t, order=2)
        pi_single, _ = project_policy(model, reward, pi0, r_t, h_single)
        td_single = compute_terminal_distribution(model, pi_single)
        occ_single = compute_occupancy(model, pi_single)
        res["terminal_kl_single_o2"] = terminal_kl(td_exact, td_single)
        res["occupancy_kl_single_o2"] = occupancy_kl(occ_exact, occ_single)

        # Atlas-direct: pick best anchor
        best_anchor, scored = select_anchor(atlas, r_t, use_geometric=True)
        res["selected_anchor"] = best_anchor.name
        res["atlas_distance"] = anchor_distance(best_anchor, r_t)

        # Atlas first-order
        pi_atlas_o1, h_atlas_o1 = transfer_from_anchor(
            best_anchor, r_t, model, reward, pi0, order=1
        )
        td_atlas_o1 = compute_terminal_distribution(model, pi_atlas_o1)
        occ_atlas_o1 = compute_occupancy(model, pi_atlas_o1)
        res["terminal_kl_atlas_o1"] = terminal_kl(td_exact, td_atlas_o1)
        res["occupancy_kl_atlas_o1"] = occupancy_kl(occ_exact, occ_atlas_o1)

        # Atlas second-order
        pi_atlas_o2, h_atlas_o2 = transfer_from_anchor(
            best_anchor, r_t, model, reward, pi0, order=2
        )
        td_atlas_o2 = compute_terminal_distribution(model, pi_atlas_o2)
        occ_atlas_o2 = compute_occupancy(model, pi_atlas_o2)
        res["terminal_kl_atlas_o2"] = terminal_kl(td_exact, td_atlas_o2)
        res["occupancy_kl_atlas_o2"] = occupancy_kl(occ_exact, occ_atlas_o2)
        res["geometry_error_atlas_o2"] = geometry_error(h_exact, h_atlas_o2)

        # Diagnostics against best anchor
        diags = compute_all_diagnostics(
            model, reward, pi0, r_t, h_exact, pi_exact, best_anchor
        )
        res.update(diags)

        all_results.append(res)

    summary = []
    for r in all_results:
        summary.append({
            "task": r["task"],
            "anchor": r["selected_anchor"],
            "d_atlas": r.get("atlas_distance", 0),
            "KL_t_single": r["terminal_kl_single_o2"],
            "KL_t_atlas_o1": r["terminal_kl_atlas_o1"],
            "KL_t_atlas_o2": r["terminal_kl_atlas_o2"],
            "KL_occ_atlas": r["occupancy_kl_atlas_o2"],
        })
    print_results_table(summary, title=f"{env_name} / {variant} - Atlas")

    return {
        "env": env_name, "variant": variant, "horizon": H,
        "n_anchors": len(atlas),
        "atlas_build_time": t_atlas,
        "tasks": all_results,
    }


def main():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

    config_files = sorted(Path(configs_dir).glob("*.yaml"))
    for cfg_path in config_files:
        try:
            result = run_config(str(cfg_path))
            save_results(
                result,
                os.path.join(outputs_dir, f"atlas_{result['variant']}.json")
            )
        except Exception as e:
            console.print(f"[red]  Error on {cfg_path.name}: {e}")

    console.print("\n[green bold]Stage B (Atlas) complete!")


if __name__ == "__main__":
    main()
