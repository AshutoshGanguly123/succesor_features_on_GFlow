# Q3 — Binary DAG: warm starts (zero optimization steps)

**What:** Compare terminal KL to the target-optimal distribution at **zero** gradient steps for: uniform scratch, reference-policy fine-tune, first-order analytical warm start, second-order warm start.

**Run:** `python q3_binary_dag_warm_start/run.py`

**Output:** `results/q3_binary_dag.json`

**Depends on:** `../lib/src/envs/binary_dag.py`
