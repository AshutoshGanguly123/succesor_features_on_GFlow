"""I/O utilities: config loading, result saving."""

import json
from pathlib import Path
import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
