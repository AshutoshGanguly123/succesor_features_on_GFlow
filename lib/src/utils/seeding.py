"""Reproducibility utilities."""

import os
import numpy as np


def set_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def set_cpu_threading(n_threads: int = 1) -> None:
    """Pin per-process threads to avoid oversubscription with multiprocessing."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
