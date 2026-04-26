#!/usr/bin/env python3
"""Section 4.5: Gymnasium tabular benchmarks (FrozenLake, CliffWalking, Taxi)."""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "lib"))

from src.experiments.run_direct_transfer import run_config
from src.utils.io import save_results


def main() -> None:
    config_dir = REPO / "lib" / "configs"
    out_dir = HERE / "results"
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
