"""Utility functions: config loading, logging, parallel execution, plotting."""

import os
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table


console = Console()

NUM_CORES = os.cpu_count() or 4
DEFAULT_WORKERS = max(1, NUM_CORES - 2)


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_path: str) -> None:
    """Save experiment results as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)


def parallel_map(fn: Callable, args_list: list, n_workers: int = None,
                 desc: str = "Processing") -> list:
    """Run fn over args_list using multiprocessing for CPU utilization.

    Uses fork-safe spawn context on macOS.
    """
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    n_workers = min(n_workers, len(args_list))
    if n_workers <= 1:
        return [fn(args) for args in args_list]

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(fn, args_list)
    return results


def timed(fn: Callable, *args, **kwargs) -> tuple:
    """Run fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def print_results_table(results: list[dict], title: str = "Results"):
    """Pretty-print results as a rich table."""
    if not results:
        return
    table = Table(title=title)
    keys = list(results[0].keys())
    for k in keys:
        table.add_column(k, justify="right")
    for row in results:
        table.add_row(*[f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])
                        for k in keys])
    console.print(table)


def set_cpu_threading(n_threads: int = 1) -> None:
    """Set environment variables to limit per-process threading.

    Call this before importing numpy/scipy in worker processes to prevent
    thread oversubscription when using multiprocessing.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
