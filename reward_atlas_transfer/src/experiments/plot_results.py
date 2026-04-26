#!/usr/bin/env python3
"""Generate all plots from PRD Section 14."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.io import load_results
from src.utils.logging import console


FIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")


def _ensure_dir():
    Path(FIGS_DIR).mkdir(parents=True, exist_ok=True)


def plot_terminal_kl_vs_distance(results: dict, env_label: str):
    """Plot A: terminal KL vs reward distance for all methods."""
    tasks = results.get("tasks", [])
    if not tasks:
        return

    norms = [t.get("delta_norm", t.get("atlas_distance", 0)) for t in tasks]
    kl_naive = [t.get("terminal_kl_naive", float("nan")) for t in tasks]
    kl_o1 = [t.get("terminal_kl_single_o1", t.get("terminal_kl_atlas_o1", float("nan"))) for t in tasks]
    kl_o2 = [t.get("terminal_kl_single_o2", t.get("terminal_kl_atlas_o2", float("nan"))) for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(norms, kl_naive, marker="^", c="#95a5a6", label="Naive reuse", alpha=0.8)
    ax.scatter(norms, kl_o1, marker="o", c="#e74c3c", label="1st-order", alpha=0.8)
    ax.scatter(norms, kl_o2, marker="s", c="#2ecc71", label="2nd-order", alpha=0.8)
    ax.set_xlabel("Reward distance ||δ||")
    ax.set_ylabel("Terminal KL")
    ax.set_title(f"{env_label}: Terminal KL vs Reward Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    valid = [v for v in kl_naive + kl_o1 + kl_o2 if np.isfinite(v) and v > 0]
    if valid:
        ax.set_yscale("symlog", linthresh=1e-10)

    path = os.path.join(FIGS_DIR, f"terminal_kl_vs_distance_{env_label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved {path}")


def plot_direct_vs_continuation(direct_results: dict, cont_results: dict, env_label: str):
    """Plot C: direct vs continuation comparison."""
    if not cont_results or not cont_results.get("tasks"):
        return

    tasks_cont = cont_results["tasks"]
    names = [t["task"] for t in tasks_cont]
    kl_direct = [t.get("terminal_kl_direct_o2", float("nan")) for t in tasks_cont]
    kl_cont = [t.get("terminal_kl_cont", float("nan")) for t in tasks_cont]
    steps = [t.get("continuation_steps", 0) for t in tasks_cont]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, kl_direct, width, label="Direct 2nd-order", color="#e74c3c", alpha=0.8)
    ax1.bar(x + width/2, kl_cont, width, label="Continuation", color="#2ecc71", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Terminal KL")
    ax1.set_title(f"{env_label}: Direct vs Continuation")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, steps, color="#3498db", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Continuation steps")
    ax2.set_title(f"{env_label}: Continuation Path Length")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, f"direct_vs_continuation_{env_label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved {path}")


def plot_taxi_horizon_sweep(results: dict):
    """Plot taxi stability across horizons."""
    horizons_data = results.get("horizons", [])
    if not horizons_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    Hs = []
    solve_times = []
    kl_vals = []
    ep_lens = []

    for hd in horizons_data:
        H = hd["horizon"]
        for task in hd["tasks"]:
            Hs.append(H)
            solve_times.append(task["exact_solve_time"])
            kl_vals.append(task.get("terminal_kl_o2", float("nan")))
            ep_lens.append(task.get("mean_episode_length", 0))

    Hs = np.array(Hs)
    solve_times = np.array(solve_times)
    kl_vals = np.array(kl_vals)
    ep_lens = np.array(ep_lens)

    for H_val in sorted(set(Hs)):
        mask = Hs == H_val
        axes[0].scatter(np.full(mask.sum(), H_val), solve_times[mask], alpha=0.7, label=f"H={H_val}")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Exact solve time (s)")
    axes[0].set_title("Solve Time vs Horizon")
    axes[0].legend()

    for H_val in sorted(set(Hs)):
        mask = Hs == H_val
        valid = np.isfinite(kl_vals[mask])
        axes[1].scatter(np.full(valid.sum(), H_val), kl_vals[mask][valid], alpha=0.7)
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Terminal KL (2nd-order)")
    axes[1].set_title("Transfer Quality vs Horizon")

    for H_val in sorted(set(Hs)):
        mask = Hs == H_val
        axes[2].scatter(np.full(mask.sum(), H_val), ep_lens[mask], alpha=0.7)
    axes[2].set_xlabel("Horizon")
    axes[2].set_ylabel("Mean episode length")
    axes[2].set_title("Episode Length vs Horizon")

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "taxi_horizon_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Saved {path}")


def main():
    _ensure_dir()
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

    # Plot direct transfer results
    for f in sorted(Path(outputs_dir).glob("direct_*.json")):
        console.print(f"  Processing {f.name}")
        data = load_results(str(f))
        label = data.get("variant", f.stem)
        plot_terminal_kl_vs_distance(data, label)

    # Plot direct vs continuation
    for f in sorted(Path(outputs_dir).glob("continuation_*.json")):
        variant = f.stem.replace("continuation_", "")
        direct_f = Path(outputs_dir) / f"direct_{variant}.json"
        cont_data = load_results(str(f))
        direct_data = load_results(str(direct_f)) if direct_f.exists() else {}
        plot_direct_vs_continuation(direct_data, cont_data, variant)

    # Plot taxi horizon sweep
    taxi_sweep = Path(outputs_dir) / "taxi_horizon_sweep.json"
    if taxi_sweep.exists():
        plot_taxi_horizon_sweep(load_results(str(taxi_sweep)))

    console.print("\n[green bold]All plots generated!")


if __name__ == "__main__":
    main()
