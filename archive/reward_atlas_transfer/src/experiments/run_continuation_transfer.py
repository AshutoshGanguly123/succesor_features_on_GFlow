#!/usr/bin/env python3
"""Stage C: Continuation transfer experiments.

For targets outside trust, perform safe multi-hop continuation.
Compare direct jump vs continuation.
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
from src.core.atlas import build_atlas, select_anchor, anchor_distance
from src.core.continuation import continuation_transfer
from src.core.occupancy import compute_terminal_distribution, compute_occupancy
from src.core.metrics import terminal_kl, occupancy_kl
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

    console.rule(f"[bold blue]{env_name} / {variant} - Continuation Transfer")

    model = extract_tabular_model(env_name, H, **cfg.get("env_kwargs", {}))
    model.validate()
    builder = ENV_FEATURE_BUILDERS[env_name]
    reward = builder(model, cfg)
    pi0 = uniform_policy(model)

    trust_r = cfg.get("trust_radius", 2.0)
    atlas, t_atlas = timed(build_atlas, cfg["anchors"], model, reward, pi0, trust_r)
    console.print(f"  Atlas: {len(atlas)} anchors in {t_atlas:.3f}s")

    all_results = []
    for task in cfg["tasks"]:
        r_t = np.array(task["r"], dtype=np.float64)
        res = {"task": task["name"], "r": r_t.tolist()}

        # Oracle
        h_exact = solve_h(model, reward, pi0, r_t)
        pi_exact, _ = project_policy(model, reward, pi0, r_t, h_exact)
        td_exact = compute_terminal_distribution(model, pi_exact)
        occ_exact = compute_occupancy(model, pi_exact)

        # Atlas-direct second-order
        best_anchor, _ = select_anchor(atlas, r_t)
        h_direct = approx_h(best_anchor, r_t, order=2)
        pi_direct, _ = project_policy(model, reward, pi0, r_t, h_direct)
        td_direct = compute_terminal_distribution(model, pi_direct)
        occ_direct = compute_occupancy(model, pi_direct)

        res["selected_anchor"] = best_anchor.name
        res["atlas_distance"] = anchor_distance(best_anchor, r_t)
        res["terminal_kl_direct_o2"] = terminal_kl(td_exact, td_direct)
        res["occupancy_kl_direct_o2"] = occupancy_kl(occ_exact, occ_direct)

        # Continuation from best anchor
        cont_result, t_cont = timed(
            continuation_transfer, best_anchor, r_t, model, reward, pi0
        )
        pi_cont = cont_result["policy"]
        td_cont = compute_terminal_distribution(model, pi_cont)
        occ_cont = compute_occupancy(model, pi_cont)

        res["terminal_kl_cont"] = terminal_kl(td_exact, td_cont)
        res["occupancy_kl_cont"] = occupancy_kl(occ_exact, occ_cont)
        res["continuation_steps"] = cont_result["path_length"]
        res["continuation_time"] = t_cont

        all_results.append(res)

    summary = []
    for r in all_results:
        summary.append({
            "task": r["task"],
            "anchor": r["selected_anchor"],
            "d_atlas": r["atlas_distance"],
            "KL_t_direct": r["terminal_kl_direct_o2"],
            "KL_t_cont": r["terminal_kl_cont"],
            "cont_steps": r["continuation_steps"],
            "t_cont": r["continuation_time"],
        })
    print_results_table(summary, title=f"{env_name} / {variant} - Continuation")

    return {
        "env": env_name, "variant": variant, "horizon": H,
        "tasks": all_results,
    }


def main():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

    # Run on smaller configs first (skip large Taxi for speed)
    config_files = sorted(Path(configs_dir).glob("*.yaml"))
    for cfg_path in config_files:
        name = cfg_path.stem
        if "taxi_h80" in name:
            console.print(f"  [yellow]Skipping {name} (too slow for continuation)")
            continue
        try:
            result = run_config(str(cfg_path))
            save_results(
                result,
                os.path.join(outputs_dir, f"continuation_{result['variant']}.json")
            )
        except Exception as e:
            console.print(f"[red]  Error on {cfg_path.name}: {e}")

    console.print("\n[green bold]Stage C (Continuation) complete!")


if __name__ == "__main__":
    main()
