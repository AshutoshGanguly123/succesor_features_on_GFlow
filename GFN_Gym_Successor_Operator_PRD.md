# PRD: Successor Operator / Bridge-Projection Transfer on Gymnasium Environments

## 1. Objective

Build a rigorous, reproducible research codebase that tests this hypothesis:

> **Do not relearn a new policy from scratch for every reward.**
> **Store how future possibilities propagate through the environment, and let reward changes tilt those possibilities.**

We will test this on **Gymnasium** tabular environments first, because Gymnasium is the maintained fork of OpenAI Gym, and the Toy Text environments are specifically designed for simple, debuggable RL experiments. Recommended first envs: **FrozenLake-v1**, **CliffWalking-v1**, and **Taxi-v3**.

---

## 2. Product Goal

### Primary goal
Implement and evaluate a **successor-operator / bridge-projection** method for reward transfer in episodic tabular environments.

### Core questions
1. Can we reuse a precomputed stochastic object instead of solving each new reward from scratch?
2. How well does this work in deterministic environments?
3. What breaks in stochastic environments?
4. Does **bridge-style projection** reduce the realizability gap?
5. Is **local free-energy geometry** useful for fast zero-shot adaptation?

### Success criteria
- Deterministic environments: near-exact zero-shot transfer.
- Stochastic environments: strong improvement over naive policy reuse.
- Second-order local geometry beats first-order for medium reward shifts.
- Clear quantitative story using standard envs.

---

## 3. Scope

### In scope
- Gymnasium Toy Text envs:
  - `FrozenLake-v1` with `is_slippery=False/True` and map sizes 4x4, 8x8.
  - `CliffWalking-v1` with `is_slippery=False/True`.
  - `Taxi-v3`.
- Exact tabular model extraction from `env.unwrapped.P`.
- Model-based dynamic programming / linear-system solves.
- Zero-shot transfer, geometry approximations, stochastic projection.
- CPU-only experiments on macOS M1.

### Out of scope for v1
- Continuous control.
- Deep neural approximation.
- Atari / MuJoCo.
- Offline datasets.

---

## 4. High-level Theory

We work with a finite episodic MDP / environment:

- states: `S`
- actions: `A`
- transition kernel: `P(s' | s, a)`
- initial distribution: `rho0(s)`
- terminal set: `T ⊂ S`

Choose a **passive reference policy** `pi0(a | s)`.
This induces a passive path measure over trajectories.

For a trajectory

`tau = (s0, a0, s1, a1, ..., s_tau)`

with terminal state `s_tau ∈ T`, define the passive path probability

```math
p_0(\tau)
= \rho_0(s_0) \prod_{t=0}^{\tau-1} \pi_0(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t).
```

Now define a task / reward through a **path score**:

```math
G_w(\tau)
= \sum_{t=0}^{\tau-1} w^\top \psi(s_t, a_t, s_{t+1}) + w^\top \beta(s_\tau).
```

- `psi` = step feature vector
- `beta` = terminal feature vector
- `w` = task weights

Define the **reward-tilted path law**:

```math
q_w(\tau)
= \frac{p_0(\tau) \exp(G_w(\tau))}{Z_w},
\qquad
Z_w = \mathbb{E}_{p_0}[\exp(G_w(\tau))].
```

This is the core probabilistic object.

---

## 5. Successor Mass / Desirability Function

Define the **successor mass** (also called a path-space partition-to-go or desirability function)

```math
h_w(s)
= \mathbb{E}_{p_0}\big[\exp(G_w(\tau_{t:})) \mid s_t = s\big].
```

### Terminal condition
For terminal `x ∈ T`:

```math
h_w(x) = \exp(w^\top \beta(x)).
```

### Nonterminal recursion
For `s ∉ T`:

```math
h_w(s)
= \sum_a \pi_0(a \mid s)
   \sum_{s'} P(s' \mid s,a)
   \exp\big(w^\top \psi(s,a,s')\big)
   h_w(s').
```

This is the exact discrete **Feynman-Kac / harmonic recursion**.

### Matrix form
Let `N` be nonterminal states and `T` terminal states.
Then

```math
h_N = M_w h_N + c_w,
\qquad
(I - M_w) h_N = c_w,
```

where

```math
M_w[s,s'']
= \sum_a \pi_0(a\mid s) P(s''\mid s,a)
  \exp(w^\top \psi(s,a,s''))
```
for `s,s'' ∈ N`, and

```math
c_w[s]
= \sum_a \pi_0(a\mid s)
  \sum_{x \in T} P(x\mid s,a)
  \exp(w^\top \psi(s,a,x)) \exp(w^\top \beta(x)).
```

This linear system is the exact tabular solve.

---

## 6. Gold-standard Tilted One-step Law

Given `h_w`, the ideal tilted **joint action-next-state law** at state `s` is

```math
q_w(a,s' \mid s)
= \frac{\pi_0(a\mid s) P(s'\mid s,a)
        \exp(w^\top \psi(s,a,s')) h_w(s')}{h_w(s)}.
```

This is the true one-step object induced by the tilted path measure.

Important: in a stochastic environment, this generally **does not** factor as

```math
\pi(a\mid s) P(s'\mid s,a)
```

for any policy `pi`, because the tilt also changes the conditional distribution of `s'` after fixing `a`.
That is the **realizability gap**.

---

## 7. Policy Projection for Stochastic Environments

We cannot control the transition kernel directly in Gymnasium. We only choose actions.
So we project the ideal tilted law back into the realizable family.

### Recommended projection
Choose the policy that minimizes the local joint KL:

```math
\pi_w^{\text{proj}}(\cdot\mid s)
= \arg\min_\pi D_{KL}
\big(q_w(a,s'\mid s) \;\|\; \pi(a\mid s) P(s'\mid s,a)\big).
```

This has a **closed-form solution**:

```math
\pi_w^{\text{proj}}(a\mid s)
= \sum_{s'} q_w(a,s'\mid s)
= \frac{\pi_0(a\mid s) Z_w(s,a)}{h_w(s)},
```

where

```math
Z_w(s,a)
= \sum_{s'} P(s'\mid s,a)
  \exp(w^\top \psi(s,a,s')) h_w(s').
```

This is the best local policy approximation in KL under fixed environment dynamics.

### Key interpretation
- deterministic envs: often exact or nearly exact
- stochastic envs: this is the best action-level approximation to the ideal tilted law

---

## 8. Local Free-energy Geometry

Define

```math
A_w(s) = \log h_w(s).
```

This is the local free energy.

If the total feature vector of the future trajectory is `Phi`, then

```math
\nabla_w A_w(s) = \mathbb{E}_{q_w}[\Phi \mid s],
```

and

```math
\nabla_w^2 A_w(s) = \mathrm{Cov}_{q_w}(\Phi \mid s).
```

So:
- first derivative = expected future features
- second derivative = curvature / how the future cloud bends under reward change

### First-order adaptation
For `w' = w + delta`:

```math
A_{w'}(s) \approx A_w(s) + g_w(s)^\top \delta,
```

where `g_w(s) = ∇_w A_w(s)`.

### Second-order adaptation

```math
A_{w'}(s)
\approx A_w(s) + g_w(s)^\top \delta
+ \frac{1}{2} \delta^\top H_w(s) \delta,
```

where `H_w(s) = ∇_w^2 A_w(s)`.

Then

```math
\hat h_{w'}(s) = \exp(\hat A_{w'}(s)),
```

and we plug `hat h` into the same projected policy formula.

---

## 9. Exact Recursions for First and Second Moments

We want exact tabular gradients / Hessians without autodiff.

Define:

```math
U_i(s) = \mathbb{E}_{p_0}[\Phi_i \exp(w^\top \Phi) \mid s],
```

```math
V_{ij}(s) = \mathbb{E}_{p_0}[\Phi_i \Phi_j \exp(w^\top \Phi) \mid s].
```

Then

```math
g_i(s) = \frac{U_i(s)}{h_w(s)},
```

```math
H_{ij}(s) = \frac{V_{ij}(s)}{h_w(s)} - g_i(s) g_j(s).
```

### Terminal conditions
For terminal `x`:

```math
U_i(x) = \beta_i(x) \exp(w^\top \beta(x)),
```

```math
V_{ij}(x) = \beta_i(x) \beta_j(x) \exp(w^\top \beta(x)).
```

### Recursion for first moments
For nonterminal `s`:

```math
U_i(s)
= \sum_a \pi_0(a\mid s) \sum_{s'} P(s'\mid s,a)
  \exp(w^\top \psi(s,a,s'))
  \Big(U_i(s') + \psi_i(s,a,s') h_w(s')\Big).
```

### Recursion for second moments

```math
V_{ij}(s)
= \sum_a \pi_0(a\mid s) \sum_{s'} P(s'\mid s,a)
  \exp(w^\top \psi(s,a,s'))
  \Big(
      V_{ij}(s')
      + \psi_i(s,a,s') U_j(s')
      + \psi_j(s,a,s') U_i(s')
      + \psi_i(s,a,s')\psi_j(s,a,s') h_w(s')
    \Big).
```

These can also be solved as linear systems.

---

## 10. Task Family Design

We need a family of rewards that changes across tasks but keeps the environment fixed.

## 10.1 FrozenLake-v1
Use features such as:
- `goal_reached`
- `fell_in_hole`
- `step_cost` (typically `-1` per step encoded as feature `steps`)
- optional direction-bias features (`left`, `right`, `up`, `down` counts)

Recommended score:

```math
G_w(\tau)
= w_1 \cdot \mathbf{1}[goal]
+ w_2 \cdot \mathbf{1}[hole]
+ w_3 \cdot steps.
```

## 10.2 CliffWalking-v1
Use features:
- `goal_reached`
- `cliff_hit_count`
- `steps`
- optional `slip_count` if slippery variant used

```math
G_w(\tau)
= w_1 \cdot \mathbf{1}[goal]
+ w_2 \cdot cliff\_hits
+ w_3 \cdot steps.
```

## 10.3 Taxi-v3
Use features:
- `successful_dropoff`
- `illegal_action_count`
- `pickup_count`
- `steps`

```math
G_w(\tau)
= w_1 \cdot successful\_dropoff
+ w_2 \cdot illegal\_actions
+ w_3 \cdot pickup\_count
+ w_4 \cdot steps.
```

These features should be positive/negative enough to create meaningful transfer tasks, but bounded so the exponential tilt stays numerically stable.

---

## 11. Experimental Design

## 11.1 Baselines
1. **Scratch solve**
   - solve each task from scratch using exact tabular dynamic programming / linear system
2. **Successor operator exact reuse**
   - compute `h_w` exactly for each target task using the operator or basis re-use
3. **First-order local geometry**
   - use `A`, `g`
4. **Second-order local geometry**
   - use `A`, `g`, `H`
5. **Naive policy reuse baseline**
   - reuse policy from nearest training task without bridge projection
6. **Optional reward-conditioned MLP baseline**
   - only if you want a black-box comparator

## 11.2 Metrics
- **Terminal KL**: KL between target terminal law and realized terminal law under projected policy
- **Occupancy KL**: KL between target state occupancy and realized occupancy
- **Expected score gap**: `E_q[G_w] - E_pi[G_w]`
- **Zero-shot runtime** per task
- **Amortized runtime** over many tasks
- **Episodes / rollouts to threshold** for online fine-tuning experiments
- **Realizability gap**:

```math
\mathcal{R}(s)
= D_{KL}\big(q_w(a,s'\mid s) \;\|\; \pi_w^{proj}(a\mid s) P(s'\mid s,a)\big)
```

- **Geometry error**:

```math
\|A_{w+\delta} - \hat A^{(1)}_{w+\delta}\|,
\qquad
\|A_{w+\delta} - \hat A^{(2)}_{w+\delta}\|
```

## 11.3 Plots
- KL vs reward shift norm
- runtime vs number of tasks
- realizability gap heatmaps over states
- success rate / average return by method
- first-order vs second-order adaptation curves

---

## 12. Concrete Milestones

### Milestone 1: exact tabular engine
- parse env model from Gymnasium
- solve `h_w` exactly
- sample from projected policy
- evaluate terminal distributions

### Milestone 2: stochastic bridge projection
- implement `q_w(a,s'|s)`
- implement closed-form `pi_proj`
- compare deterministic vs stochastic envs

### Milestone 3: local geometry
- compute `U`, `V`
- first-order and second-order zero-shot transfer
- sweep reward shifts

### Milestone 4: serious benchmark report
- FrozenLake 4x4 / 8x8
- CliffWalking
- Taxi
- aggregate plots and markdown report

---

## 13. Recommended Repo Layout

```text
successor-operator-gym/
  README.md
  pyproject.toml
  requirements.txt
  src/
    successor_operator/
      __init__.py
      mdp.py
      gym_extract.py
      passive_policy.py
      features.py
      feynman_kac.py
      moments.py
      projection.py
      rollout.py
      metrics.py
      utils.py
  experiments/
    run_frozenlake.py
    run_cliffwalking.py
    run_taxi.py
    sweep_geometry.py
    sweep_transfer.py
  configs/
    frozenlake.yaml
    cliffwalking.yaml
    taxi.yaml
  tests/
    test_feynman_kac.py
    test_projection.py
    test_moments.py
    test_gym_extract.py
  outputs/
    figures/
    tables/
    logs/
```

---

## 14. Environment and Tooling

Use **Gymnasium**, not old `gym`, because Gymnasium is the maintained fork of OpenAI Gym and provides a migration path for older Gym code.

### macOS M1 setup
Recommended:
- Python 3.11
- `uv` or `poetry`
- CPU only
- `numpy`, `scipy`, `gymnasium`, `matplotlib`, `pandas`, `tqdm`, `pytest`

### Install
```bash
uv venv .venv
source .venv/bin/activate
uv pip install gymnasium numpy scipy matplotlib pandas tqdm pyyaml pytest rich
```

### CPU notes for 8-core M1
Set:
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```
Then parallelize experiments over tasks with 4 to 6 worker processes.

---

## 15. Python Interfaces

## 15.1 Core data structure

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TabularMDP:
    nS: int
    nA: int
    P: np.ndarray          # shape [nS, nA, nS]
    terminal: np.ndarray   # shape [nS], bool
    start: np.ndarray      # shape [nS]
    legal: np.ndarray      # shape [nS, nA], bool
```

## 15.2 Gymnasium model extraction

```python
import gymnasium as gym
import numpy as np


def extract_tabular_mdp(env_name: str, **kwargs) -> TabularMDP:
    env = gym.make(env_name, **kwargs)
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise ValueError(f"{env_name} does not expose a tabular transition model")

    nS = base.observation_space.n
    nA = base.action_space.n
    P = np.zeros((nS, nA, nS), dtype=np.float64)
    terminal = np.zeros(nS, dtype=bool)
    legal = np.ones((nS, nA), dtype=bool)

    for s in range(nS):
        for a in range(nA):
            outcomes = base.P[s][a]
            if len(outcomes) == 0:
                legal[s, a] = False
                continue
            for prob, s_next, reward, done in outcomes:
                P[s, a, s_next] += prob
                if done:
                    terminal[s_next] = True

    start = np.zeros(nS, dtype=np.float64)
    if hasattr(base, "initial_state_distrib"):
        start[:] = np.asarray(base.initial_state_distrib, dtype=np.float64)
    else:
        # fallback: deterministic reset sample once
        obs, _ = env.reset(seed=0)
        start[int(obs)] = 1.0

    env.close()
    return TabularMDP(nS=nS, nA=nA, P=P, terminal=terminal, start=start, legal=legal)
```

## 15.3 Passive policy

```python
import numpy as np


def uniform_passive_policy(mdp: TabularMDP) -> np.ndarray:
    pi0 = mdp.legal.astype(np.float64)
    row_sums = pi0.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return pi0 / row_sums
```

## 15.4 Feature functions

```python
import numpy as np


def frozenlake_features(mdp: TabularMDP, env_desc: np.ndarray):
    # returns step_feat[s, a, s', d], terminal_feat[s, d]
    nS, nA = mdp.nS, mdp.nA
    d = 3  # goal, hole, step
    step_feat = np.zeros((nS, nA, nS, d), dtype=np.float64)
    term_feat = np.zeros((nS, d), dtype=np.float64)

    # step cost as +1 count, user can make weight negative
    step_feat[..., 2] = 1.0

    flat = env_desc.flatten()
    for s in range(nS):
        c = chr(flat[s]) if isinstance(flat[s], (np.uint8, int)) else str(flat[s])
        if c == "G":
            term_feat[s, 0] = 1.0
        elif c == "H":
            term_feat[s, 1] = 1.0
    return step_feat, term_feat
```

## 15.5 Exact successor-mass solve

```python
import numpy as np
from scipy.linalg import solve


def solve_successor_mass(
    mdp: TabularMDP,
    pi0: np.ndarray,
    step_feat: np.ndarray,
    term_feat: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    nS = mdp.nS
    nonterm = np.where(~mdp.terminal)[0]
    term = np.where(mdp.terminal)[0]

    h = np.zeros(nS, dtype=np.float64)
    h[term] = np.exp(term_feat[term] @ w)

    if len(nonterm) == 0:
        return h

    idx = {s: i for i, s in enumerate(nonterm)}
    M = np.zeros((len(nonterm), len(nonterm)), dtype=np.float64)
    c = np.zeros(len(nonterm), dtype=np.float64)

    for s in nonterm:
        i = idx[s]
        for a in range(mdp.nA):
            pa = pi0[s, a]
            if pa == 0.0:
                continue
            for s2 in range(nS):
                p = mdp.P[s, a, s2]
                if p == 0.0:
                    continue
                weight = pa * p * np.exp(step_feat[s, a, s2] @ w)
                if mdp.terminal[s2]:
                    c[i] += weight * h[s2]
                else:
                    M[i, idx[s2]] += weight

    h_non = solve(np.eye(len(nonterm)) - M, c, assume_a="gen")
    h[nonterm] = h_non
    return h
```

## 15.6 First and second moments

```python
import numpy as np
from scipy.linalg import solve


def solve_moments(mdp, pi0, step_feat, term_feat, w, h):
    nS = mdp.nS
    d = step_feat.shape[-1]
    nonterm = np.where(~mdp.terminal)[0]
    term = np.where(mdp.terminal)[0]
    idx = {s: i for i, s in enumerate(nonterm)}

    # Build shared linear operator
    M = np.zeros((len(nonterm), len(nonterm)), dtype=np.float64)
    for s in nonterm:
        i = idx[s]
        for a in range(mdp.nA):
            pa = pi0[s, a]
            if pa == 0.0:
                continue
            for s2 in nonterm:
                p = mdp.P[s, a, s2]
                if p == 0.0:
                    continue
                M[i, idx[s2]] += pa * p * np.exp(step_feat[s, a, s2] @ w)

    A = np.eye(len(nonterm)) - M

    U = np.zeros((nS, d), dtype=np.float64)
    V = np.zeros((nS, d, d), dtype=np.float64)

    if len(term) > 0:
        term_exp = np.exp(term_feat[term] @ w)
        U[term] = term_feat[term] * term_exp[:, None]
        for k, s in enumerate(term):
            beta = term_feat[s]
            V[s] = np.outer(beta, beta) * np.exp(beta @ w)

    # Solve U dimension by dimension
    for dim in range(d):
        b = np.zeros(len(nonterm), dtype=np.float64)
        for s in nonterm:
            i = idx[s]
            for a in range(mdp.nA):
                pa = pi0[s, a]
                if pa == 0.0:
                    continue
                for s2 in range(nS):
                    p = mdp.P[s, a, s2]
                    if p == 0.0:
                        continue
                    psi = step_feat[s, a, s2, dim]
                    weight = pa * p * np.exp(step_feat[s, a, s2] @ w)
                    cont = psi * h[s2] + U[s2, dim]
                    b[i] += weight * cont
        U[nonterm, dim] = solve(A, b, assume_a="gen")

    # Solve V dimension by dimension
    for i_dim in range(d):
        for j_dim in range(d):
            b = np.zeros(len(nonterm), dtype=np.float64)
            for s in nonterm:
                i = idx[s]
                for a in range(mdp.nA):
                    pa = pi0[s, a]
                    if pa == 0.0:
                        continue
                    for s2 in range(nS):
                        p = mdp.P[s, a, s2]
                        if p == 0.0:
                            continue
                        psi_i = step_feat[s, a, s2, i_dim]
                        psi_j = step_feat[s, a, s2, j_dim]
                        weight = pa * p * np.exp(step_feat[s, a, s2] @ w)
                        cont = (
                            V[s2, i_dim, j_dim]
                            + psi_i * U[s2, j_dim]
                            + psi_j * U[s2, i_dim]
                            + psi_i * psi_j * h[s2]
                        )
                        b[i] += weight * cont
            V[nonterm, i_dim, j_dim] = solve(A, b, assume_a="gen")

    grad = U / np.maximum(h[:, None], 1e-300)
    hess = V / np.maximum(h[:, None, None], 1e-300) - grad[:, :, None] * grad[:, None, :]
    return grad, hess
```

## 15.7 Projected policy

```python
import numpy as np


def projected_policy(mdp, pi0, step_feat, w, h):
    nS, nA = mdp.nS, mdp.nA
    pi = np.zeros((nS, nA), dtype=np.float64)

    for s in range(nS):
        if mdp.terminal[s]:
            continue
        z = np.zeros(nA, dtype=np.float64)
        for a in range(nA):
            if pi0[s, a] == 0.0:
                continue
            total = 0.0
            for s2 in range(nS):
                p = mdp.P[s, a, s2]
                if p == 0.0:
                    continue
                total += p * np.exp(step_feat[s, a, s2] @ w) * h[s2]
            z[a] = pi0[s, a] * total
        zsum = z.sum()
        if zsum > 0.0:
            pi[s] = z / zsum
        else:
            pi[s] = pi0[s]
    return pi
```

## 15.8 First/second-order zero-shot approximation

```python
import numpy as np


def approximate_h_from_geometry(A, grad, hess, delta, order=2):
    approx = A + grad @ delta
    if order >= 2:
        quad = np.einsum("bi,ij,bj->b", hess, np.eye(len(delta)), np.zeros_like(hess[:,0,:]))
        # easier and clearer:
        quad = np.einsum("bi,bij,bj->b", delta[None, :], hess, delta[None, :])
        approx = approx + 0.5 * quad
    return np.exp(approx)
```

Use the following corrected version in actual code:

```python
def approximate_h_from_geometry(A, grad, hess, delta, order=2):
    approx = A + grad @ delta
    if order >= 2:
        quad = np.einsum("i,sij,j->s", delta, hess, delta)
        approx = approx + 0.5 * quad
    return np.exp(approx)
```

## 15.9 Rollout evaluation

```python
import numpy as np


def sample_episode(env, pi, max_steps=1000, seed=None):
    obs, info = env.reset(seed=seed)
    total_feat = None
    traj = [int(obs)]
    done = False
    t = 0
    while not done and t < max_steps:
        s = int(obs)
        a = np.random.choice(len(pi[s]), p=pi[s])
        obs2, reward, terminated, truncated, info = env.step(a)
        traj.append(int(obs2))
        obs = obs2
        done = terminated or truncated
        t += 1
    return traj
```

---

## 16. Validation Tests

### Unit tests
1. `h_w` from linear system matches Monte Carlo estimate of `E[exp(G_w)]`
2. `q_w(a,s'|s)` sums to 1 for every nonterminal state
3. `pi_proj` sums to 1 and matches action marginal of `q_w`
4. in deterministic envs, projected policy should match exact tilt up to numerical tolerance
5. `grad` and `hess` match finite differences of `A_w = log h_w`

### Numerical checks
- max probability normalization error < `1e-12`
- finite-difference gradient error < `1e-5`
- finite-difference Hessian error < `1e-4`

---

## 17. Benchmark Matrix

| Env | Variant | Why it matters |
|---|---|---|
| FrozenLake-v1 | deterministic 4x4, 8x8 | exact transfer should look very strong |
| FrozenLake-v1 | slippery 4x4, 8x8 | clean stochastic realizability-gap test |
| CliffWalking-v1 | deterministic | tests path-cost shaping and hazard avoidance |
| CliffWalking-v1 | slippery | another stochastic hazard environment |
| Taxi-v3 | default | richer state graph, loops, illegal actions |

FrozenLake, CliffWalking, and Taxi are all documented Gymnasium Toy Text environments with small discrete spaces, making them appropriate debugging / research benchmarks for tabular algorithms.

---

## 18. Reporting Template

For each env, report:
- task family definition
- passive policy
- exact solve runtime
- projected-policy runtime
- terminal KL
- expected score gap
- success rate
- average episode length
- realizability gap statistics
- first-order vs second-order errors

Then aggregate across envs.

---

## 19. Recommended First Sprint

### Sprint 1
- implement `extract_tabular_mdp`
- implement `solve_successor_mass`
- implement `projected_policy`
- reproduce deterministic/slippery FrozenLake

### Sprint 2
- add `solve_moments`
- add first-order / second-order transfer sweeps
- add CliffWalking

### Sprint 3
- add Taxi
- add benchmark tables and plots
- finalize writeup

---

## 20. Expected Research Outcomes

### Best-case
- deterministic envs are essentially solved
- stochastic envs show that bridge projection is a principled and useful fix
- local free-energy geometry gives fast approximate transfer

### Most valuable failure mode
- if stochastic envs still fail badly, we get a very clear statement of the limitation:
  **the gap is not in learning, it is in realizability under fixed action-controlled dynamics**

That is still a publishable insight if demonstrated cleanly.

---

## 21. Bottom-line Implementation Philosophy

1. Start with **exact tabular Gymnasium environments**.
2. Treat reward transfer as **change of measure on path space**.
3. Compute the exact **successor mass** `h_w`.
4. Use the exact tilted one-step law as the gold standard.
5. Project to the best realizable policy when transitions are stochastic.
6. Use first and second derivatives of `log h_w` for zero-shot adaptation.
7. Only after the tabular story is solid, move to learned approximations.

That is the cleanest path to results that mean something.


## 22. References

- Gymnasium main documentation
- Gymnasium migration guide from OpenAI Gym
- Gymnasium Toy Text documentation
- FrozenLake-v1 documentation
- CliffWalking-v1 documentation
- Taxi-v3 documentation
