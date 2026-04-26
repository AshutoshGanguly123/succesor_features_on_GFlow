"""Logging and display utilities."""

import os
import time
import multiprocessing as mp
from typing import Callable

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

NUM_CORES = os.cpu_count() or 4
DEFAULT_WORKERS = max(1, NUM_CORES - 2)


def timed(fn: Callable, *args, **kwargs) -> tuple:
    """Run fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def parallel_map(fn: Callable, args_list: list,
                 n_workers: int = None, desc: str = "Processing") -> list:
    """Parallel map using multiprocessing spawn context (macOS safe)."""
    if n_workers is None:
        n_workers = DEFAULT_WORKERS
    n_workers = min(n_workers, len(args_list))
    if n_workers <= 1:
        return [fn(a) for a in args_list]
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        return pool.map(fn, args_list)


def print_results_table(rows: list[dict], title: str = "Results"):
    if not rows:
        return
    table = Table(title=title)
    keys = list(rows[0].keys())
    for k in keys:
        table.add_column(k, justify="right")
    for row in rows:
        table.add_row(*[
            f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])
            for k in keys
        ])
    console.print(table)
