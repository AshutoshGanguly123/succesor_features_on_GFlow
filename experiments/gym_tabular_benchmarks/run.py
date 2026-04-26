#!/usr/bin/env python3
"""Run report Section 4.5 Gym tabular benchmarks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reward_atlas_transfer"))

from src.experiments.run_direct_transfer import run_config
from src.utils.io import save_results


def main() -> None:
    config_dir = ROOT / "reward_atlas_transfer" / "configs"
    out_dir = ROOT / "outputs" / "gym_tabular_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in sorted(config_dir.glob("*.yaml")):
        res = run_config(str(cfg))
        results.append(res)
        variant = res["variant"]
        save_results(res, str(out_dir / f"direct_{variant}.json"))

    save_results({"runs": results}, str(out_dir / "all_results.json"))
    print(f"Saved benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()
