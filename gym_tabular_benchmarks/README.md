# Gym tabular benchmarks (Section 4.5)

**What:** Exact backward DP on time-augmented tabular models for FrozenLake (4×4 / 8×8, deterministic and slippery), CliffWalking (deterministic), Taxi (\(H \in \{20,40,80\}\)). Metrics: terminal KL naive vs first- vs second-order transfer, occupancy KL, realizability gap mean.

**Configs:** `../lib/configs/*.yaml` (do not move; paths are wired to the library).

**Run:** `python gym_tabular_benchmarks/run.py` (several minutes; Taxi + large \(H\) dominates).

**Output:** `results/direct_*.json` and `results/all_results.json`

**Depends on:** full `../lib` package (`src/core`, `src/envs`, `run_direct_transfer`).
