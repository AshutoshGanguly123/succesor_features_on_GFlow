# Successor flow features — experiments only

This repository is trimmed to **four experiment folders** (report Section 4) plus a small shared library.

| Folder | Report |
|--------|--------|
| [q1_binary_dag_linear_loglinear](q1_binary_dag_linear_loglinear/) | §4.2 Linear vs log-linear decomposition |
| [q2_binary_dag_cumulant_order](q2_binary_dag_cumulant_order/) | §4.3 First- vs second-order cumulant transfer |
| [q3_binary_dag_warm_start](q3_binary_dag_warm_start/) | §4.4 Warm-start at zero steps |
| [gym_tabular_benchmarks](gym_tabular_benchmarks/) | §4.5 FrozenLake, CliffWalking, Taxi |

**Shared code:** [lib/](lib/) — tabular DP, geometry, Gym extraction, `binary_dag.py`, tests, YAML configs.

## Setup

```bash
cd lib
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

## Run (from repository root)

```bash
python q1_binary_dag_linear_loglinear/run.py
python q2_binary_dag_cumulant_order/run.py
python q3_binary_dag_warm_start/run.py
python gym_tabular_benchmarks/run.py
```

Each script writes JSON under that folder’s `results/`. Committed snapshots are already in `results/`; re-running overwrites them.

## Tests

```bash
cd lib && pytest
```
