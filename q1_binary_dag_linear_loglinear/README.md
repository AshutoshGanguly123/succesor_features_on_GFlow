# Q1 — Binary DAG: linear exactness vs log-linear obstruction

**What:** On a complete binary tree (depth 8 → 255 internal, 256 terminals) with a fixed reference flow at \(w_0\), we check (1) **linear** terminal rewards: \(h_w = \Psi^\top w\) to machine precision; (2) **log-linear** rewards: a single global linear fit to \(\log h_w\) across many \(w\) leaves residual RMSE and terminal KL when converted to a policy.

**Run:** from repo root, `python q1_binary_dag_linear_loglinear/run.py`

**Output:** `results/q1_binary_dag.json`

**Depends on:** `../lib/src/envs/binary_dag.py` (NumPy only).
