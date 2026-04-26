# Successor flow features on GFlowNets (report-aligned code)

This repository contains a **minimal, reproducible** implementation of the empirical blocks from the course/report narrative on **successor flow features (SFFs)** for reward-conditioned flows: exact checks on a **deterministic binary construction DAG**, and **tabular Gymnasium** benchmarks (FrozenLake, CliffWalking, Taxi) with exact dynamic programming and geometry-based transfer.

Older exploratory code (duplicate Gym stack, reward-atlas and continuation drivers, internal PRDs) lives under [`archive/`](archive/) and is **not** required to reproduce the main report experiments.

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`experiments/q1_binary_dag_linear_loglinear/run.py`](experiments/q1_binary_dag_linear_loglinear/run.py) | **Q1:** linear exactness vs global log-linear obstruction (Section 4.2). |
| [`experiments/q2_binary_dag_cumulant_order/run.py`](experiments/q2_binary_dag_cumulant_order/run.py) | **Q2:** first- vs second-order cumulant expansion along \(\Delta w\) (Section 4.3). |
| [`experiments/q3_binary_dag_warm_start/run.py`](experiments/q3_binary_dag_warm_start/run.py) | **Q3:** warm-start schemes at **zero** gradient steps (Section 4.4 / Table 1 style). |
| [`experiments/gym_tabular_benchmarks/run.py`](experiments/gym_tabular_benchmarks/run.py) | **Section 4.5:** all YAML configs under `reward_atlas_transfer/configs/`. |
| [`reward_atlas_transfer/src/`](reward_atlas_transfer/src/) | Core library: tabular model extraction, backward DP for \(h\), tilted moments (\(\tilde\Psi\), \(\Sigma\)), projection, occupancy, KL metrics, realizability gap. |
| [`reward_atlas_transfer/configs/*.yaml`](reward_atlas_transfer/configs/) | Gym environments, horizons, anchor weights, and task list per variant. |
| [`reward_atlas_transfer/tests/`](reward_atlas_transfer/tests/) | Unit tests (binary DAG, DP, geometry, projection, model extraction). |
| [`findings/`](findings/) | **Committed snapshots** of numerical outputs + narrative summary ([`findings/FINDINGS.md`](findings/FINDINGS.md)). |
| [`archive/`](archive/) | Superseded or non–main-text code and assets (kept for history). |

---

## Setup

**Requirements:** Python 3.10+ recommended (3.9 used in CI-style runs is acceptable if dependencies resolve).

```bash
cd reward_atlas_transfer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For experiment scripts run **from the repository root**, `PYTHONPATH` is adjusted inside each script to include `reward_atlas_transfer/`.

---

## Running experiments

From the **repository root** (`cloud_tilting/`):

```bash
# Binary DAG (Sections 4.1–4.4)
python experiments/q1_binary_dag_linear_loglinear/run.py
python experiments/q2_binary_dag_cumulant_order/run.py
python experiments/q3_binary_dag_warm_start/run.py

# Gym tabular benchmarks (Section 4.5); full suite may take several minutes (Taxi + large H)
python experiments/gym_tabular_benchmarks/run.py
```

**Default output location (gitignored):** `outputs/binary_dag/*.json`, `outputs/gym_tabular_benchmarks/*.json`.

**Committed snapshots:** see [`findings/`](findings/) and read [`findings/FINDINGS.md`](findings/FINDINGS.md) for interpretation.

---

## Tests

```bash
cd reward_atlas_transfer
pytest
```

---

## Binary DAG model (brief)

The class [`BinaryDAG`](reward_atlas_transfer/src/envs/binary_dag.py) builds a **complete binary tree** of configurable depth (default depth **8** → 255 internal nodes, 256 terminals). Terminal feature vectors \(\phi(x) \in \mathbb{R}^d\) define:

- **Linear** terminal rewards \(R_w(x) = \phi(x)^\top w\) for Proposition 1 checks.
- **Log-linear** terminal rewards \(R_w(x) = \exp(\phi(x)^\top w)\) for log-\(h_w\) and cumulant experiments.

The **reference flow** at \(w_0\) is the **optimal** GFlowNet flow on the tree (subtree sums of terminal \(R_{w_0}\)); the forward policy is local normalization. The **descendant measure** \(\mu_s\) is the terminal hitting distribution from each node under that reference policy. All \(h_w(s)\) reported in Q1–Q3 are **exact expectations** under that \(\mu_s\) (no sampling noise).

---

## Gym pipeline (brief)

1. **Extract** a time-augmented tabular MDP from Gymnasium (`FrozenLake-v1`, `CliffWalking-v0`, `Taxi-v3`).
2. **Solve** successor mass \(h_r(s,t)\) by backward DP ([`dp.py`](reward_atlas_transfer/src/core/dp.py)).
3. **Build anchor** at reference weights \(r_{\mathrm{anchor}}\): exact \(h\), \(\nabla \log h\), \(\nabla^2 \log h\) ([`geometry.py`](reward_atlas_transfer/src/core/geometry.py)).
4. **Transfer** to new \(r\) with first- and second-order Taylor approximations, **project** to a tabular forward policy, and compare **terminal** and **occupancy** KL to the exact solve; report **realizability gap** diagnostics ([`bridge_projection.py`](reward_atlas_transfer/src/core/bridge_projection.py), [`diagnostics.py`](reward_atlas_transfer/src/core/diagnostics.py)).

---

## Citation / context

This codebase supports the empirical claims around:

1. Exact **linear** decomposition of \(h_w\) via successor features under a fixed reference flow.
2. **Obstruction** to a global **log-linear** inner-product representation of \(\log h_w\).
3. **Local** recovery via a **cumulant** (Taylor) expansion and its empirical benefit for transfer and warm starts.
4. **Gym** tabular instantiations and **stochastic realizability** effects on slippery grids.

---

## Publishing to GitHub

If this folder is not yet linked to your remote:

```bash
git remote add origin https://github.com/AshutoshGanguly123/succesor_features_on_GFlow.git
git push -u origin main
```

Use the branch you are on if it is not `main`.

## License

Add a `LICENSE` file if you intend open redistribution; none is bundled by default.
