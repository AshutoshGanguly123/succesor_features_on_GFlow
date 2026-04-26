# PRD v2: Reward-Atlas, Continuation, and Bridge Projection for Gym / Gymnasium Transfer

## 1. Thesis

We are testing a sharper version of the original idea:

> **Do not store a policy for each reward.**
> **Store how future possibility mass propagates through the environment, then let reward changes tilt that mass.**

The new upgrade is this:

> **Reward space is not one smooth global bowl. It is an atlas of local bowls.**
> **When a reward shift is small, local geometry is enough.**
> **When a reward shift crosses a route-switch boundary, we need a new local chart or a continuation path.**

This PRD turns that philosophy into a rigorous tabular research system on OpenAI Gym / Gymnasium-style environments.

---

## 2. Why v2 exists

The first round of experiments taught us three things.

1. **Deterministic environments are friendly.**
   Successor-mass style transfer and local geometry work very well.

2. **Stochastic environments expose a realizability gap.**
   The ideal reward-tilted one-step law is over `(action, next_state)`, but the agent only controls `action`.

3. **Large reward shifts are not all alike.**
   Sometimes the shift is big but smooth. Sometimes a tiny shift crosses a behavioral boundary and flips the preferred route family. In those cases, direct first/second-order geometry around the wrong anchor becomes unreliable.

So v2 adds four things:

- **finite-horizon / time-augmented solves** to make looping envs like Taxi stable,
- **an atlas of anchor rewards** with local geometry around each anchor,
- **a meta-controller over reward space** to decide which anchor to trust,
- **continuation / homotopy in reward space** when a target reward is too far or too close to a route boundary.

---

## 3. Product goal

Build a reproducible codebase that answers these questions on standard tabular RL environments:

1. Can we transfer to a new reward without solving from scratch?
2. When is direct local geometry enough?
3. Can an atlas of local models beat a single global reference?
4. Can continuation repair failures near route-switch boundaries?
5. Does bridge projection reduce the stochastic realizability gap?
6. Which diagnostics predict failure before we see it in KL curves?

---

## 4. Core hypotheses

### H1. Deterministic transfer
In deterministic episodic environments, exact successor-mass transfer with bridge projection is exact or nearly exact, and local second-order geometry should dominate naive reuse for moderate reward shifts.

### H2. Stochastic transfer
In stochastic environments, the ideal tilt is not directly realizable as a policy, but bridge projection should still yield strong transfer and much lower KL than naive reuse.

### H3. Locality matters
Second-order geometry should be better than first-order **inside a trust region**, but can overshoot outside it.

### H4. Atlas beats single anchor
A reward atlas with anchor selection should outperform direct reuse from a single reference reward.

### H5. Continuation beats direct jump near boundaries
When a target reward is near a route-switch boundary, continuation through several small reward tilts should outperform a single direct jump.

### H6. Diagnostics are predictive
Curvature, action-margin collapse, first-vs-second-order disagreement, and realizability gap should predict when direct local transfer will fail.

### H7. Taxi-like loops need time augmentation
Without horizon control, looping environments can produce unstable or non-interpretable path tilts. Time augmentation or finite horizon should repair this.

---

## 5. Environment scope

## Phase 1: exact tabular, Gym-style environments

Required:
- `FrozenLake-v1` 4x4 deterministic
- `FrozenLake-v1` 4x4 slippery
- `FrozenLake-v1` 8x8 deterministic
- `FrozenLake-v1` 8x8 slippery
- `CliffWalking-v1` deterministic
- `Taxi-v3` with explicit finite horizon via time augmentation

Optional later:
- `WindyGridworld` if available in your chosen package
- small `MiniGrid` tasks only after tabular pipeline is stable

### Important implementation note
Even if the runtime env uses wrappers, **the research pipeline should always build an explicit tabular model**:
- transition tensor `P[s, a, s']`
- terminal indicator
- finite horizon `H`
- feature map

This avoids “wrapper folklore” bugs and makes results reproducible.

---

## 6. Mathematical formulation

## 6.1 Time-augmented episodic model

We use a finite-horizon episodic MDP.

- state space: `S`
- action space: `A`
- transition kernel: `P(s' | s, a)`
- horizon: `H`
- terminal set: `T`
- initial distribution: `rho0`

To make looping environments rigorous, we augment by time:

```math
\tilde s_t = (s,t), \qquad t \in \{0,1,\dots,H\}.
```

The process stops when either:
- `s ∈ T`, or
- `t = H`.

This turns every environment into a finite DAG in time, even if the original state graph has cycles.

That single move is the antidote to the Taxi dragon.

---

## 6.2 Passive reference path measure

Choose a reference policy `pi0(a | s,t)`. Usually uniform or mildly biased but fixed.

For a trajectory

```math
\tau = (s_0,a_0,s_1,a_1,\dots,s_T)
```

its passive path probability is

```math
p_0(\tau)
= \rho_0(s_0) \prod_{t=0}^{T-1} \pi_0(a_t\mid s_t,t) P(s_{t+1}\mid s_t,a_t).
```

---

## 6.3 Reward / feature parameterization

We parameterize tasks by a feature vector.

Step features:

```math
\psi(s,a,s',t) \in \mathbb R^d
```

Terminal features:

```math
\beta(s,t) \in \mathbb R^d
```

Task weights:

```math
r \in \mathbb R^d.
```

Path score:

```math
G_r(\tau)
= \sum_{t=0}^{T-1} r^\top \psi(s_t,a_t,s_{t+1},t) + r^\top \beta(s_T,T).
```

The reward-tilted path law is

```math
q_r(\tau)
= \frac{p_0(\tau)\exp(G_r(\tau))}{Z(r)},
\qquad
Z(r) = \mathbb E_{p_0}[\exp(G_r(\tau))].
```

This is the master object. Everything else is a shadow of this.

---

## 6.4 Successor mass / desirability function

Define the successor mass:

```math
h_r(s,t)
= \mathbb E_{p_0}\big[\exp(G_r(\tau_{t:})) \mid s_t=s\big].
```

Boundary conditions:

```math
h_r(s,t) = \exp(r^\top \beta(s,t))
\quad \text{if } s\in T \text{ or } t=H.
```

Backward recursion:

```math
h_r(s,t)
= \sum_a \pi_0(a\mid s,t)
   \sum_{s'} P(s'\mid s,a)
   \exp\big(r^\top \psi(s,a,s',t)\big)
   h_r(s',t+1).
```

This is the exact Feynman-Kac / harmonic recursion.

### Matrix form at each time slice
For nonterminal states,

```math
h_t = M_t(r) h_{t+1} + c_t(r),
```

where the boundary contribution may be folded into `h_{t+1}` if terminal states are kept explicitly.

Because we are finite-horizon, backward dynamic programming is preferred over a global linear solve.

---

## 6.5 Ideal tilted one-step law

Given exact `h_r`, the ideal one-step law at `(s,t)` is

```math
q_r(a,s'\mid s,t)
= \frac{\pi_0(a\mid s,t)
        P(s'\mid s,a)
        \exp(r^\top \psi(s,a,s',t))
        h_r(s',t+1)}{h_r(s,t)}.
```

This is exact at the path-measure level.

### Important point
In stochastic environments, this joint law generally does **not** factor as

```math
\pi(a\mid s,t) P(s'\mid s,a)
```

for any policy `pi`.

That mismatch is the realizability gap.

---

## 6.6 Bridge projection back to policy space

We only control actions, not transitions. So we project the ideal tilted one-step law back to the realizable action family by solving

```math
\pi_r^{\mathrm{proj}}(\cdot\mid s,t)
= \arg\min_{\pi}
D_{KL}\Big(q_r(a,s'\mid s,t)
\;\|\; \pi(a\mid s,t)P(s'\mid s,a)\Big).
```

This has a closed form:

```math
\pi_r^{\mathrm{proj}}(a\mid s,t)
= \sum_{s'} q_r(a,s'\mid s,t)
= \frac{\pi_0(a\mid s,t) Z_r(s,a,t)}{h_r(s,t)},
```

with action score

```math
Z_r(s,a,t)
= \sum_{s'} P(s'\mid s,a)
    \exp(r^\top \psi(s,a,s',t))
    h_r(s',t+1).
```

Interpretation:
- `h_r(s,t)` = total reward-weighted future mass under the reference process.
- `Z_r(s,a,t)` = how much of that future mass opens up if you take action `a` now.
- `pi_r_proj` = normalized action-level shadow of the ideal cloud tilt.

---

## 6.7 Realizability gap

The minimum KL value of the projection is itself a useful diagnostic:

```math
\Gamma_r(s,t)
= \min_{\pi}
D_{KL}\Big(q_r(a,s'\mid s,t)
\;\|\; \pi(a\mid s,t)P(s'\mid s,a)\Big).
```

Since the minimizer is `pi_r_proj`, this becomes

```math
\Gamma_r(s,t)
= \sum_a q_r(a\mid s,t)
   D_{KL}\Big(q_r(s'\mid s,a,t) \;\|\; P(s'\mid s,a)\Big).
```

This is beautiful and interpretable.

It says:
> the realizability gap is exactly how much the tilt changes the next-state law beyond what can be achieved by action selection alone.

Aggregate metrics:

```math
\bar \Gamma_r = \sum_{t,s} d_r(s,t) \Gamma_r(s,t),
\qquad
\Gamma_r^{\max} = \max_{s,t} \Gamma_r(s,t),
```

where `d_r(s,t)` is the occupancy of the projected policy or the ideal tilted law, depending on which story you want to tell.

---

## 6.8 Local free energy and geometry

Define the local free energy:

```math
A_r(s,t) = \log h_r(s,t).
```

Because this is a log-partition object, its derivatives have probabilistic meaning.

Gradient:

```math
\mu_r(s,t)
= \nabla_r A_r(s,t)
= \mathbb E_{q_r}[\Phi_{t:} \mid s_t=s],
```

where `Phi_{t:}` is the total future feature vector from time `t` onward.

Hessian:

```math
\Sigma_r(s,t)
= \nabla_r^2 A_r(s,t)
= \mathrm{Cov}_{q_r}(\Phi_{t:} \mid s_t=s).
```

This is the local geometry. Curvature is not a metaphor here. It is covariance.

### First-order approximation at anchor `r_k`
Let `delta = r - r_k`.

```math
\widehat A^{(1)}_{k,r}(s,t)
= A_{r_k}(s,t) + \mu_{r_k}(s,t)^\top \delta.
```

### Second-order approximation

```math
\widehat A^{(2)}_{k,r}(s,t)
= A_{r_k}(s,t) + \mu_{r_k}(s,t)^\top \delta
+ \tfrac12 \delta^\top \Sigma_{r_k}(s,t) \delta.
```

Then

```math
\widehat h^{(m)}_{k,r}(s,t) = \exp\big(\widehat A^{(m)}_{k,r}(s,t)\big),
\qquad m\in\{1,2\}.
```

and we plug `hat h` into the action scores and bridge projection.

---

## 6.9 Exact moment recursions at an anchor

At an anchor reward `r_k`, after solving `h_k = h_{r_k}`, define the exact tilted joint law

```math
q_k(a,s'\mid s,t)
= \frac{\pi_0(a\mid s,t)P(s'\mid s,a)e^{r_k^\top \psi(s,a,s',t)}h_k(s',t+1)}{h_k(s,t)}.
```

### Mean feature recursion
Boundary:

```math
\mu_k(s,t) = \beta(s,t)
\quad \text{if } s\in T \text{ or } t=H.
```

Backward recursion:

```math
\mu_k(s,t)
= \sum_{a,s'} q_k(a,s'\mid s,t)
  \big(\psi(s,a,s',t) + \mu_k(s',t+1)\big).
```

### Uncentered second moment recursion
Define `S_k(s,t) = E[Phi Phi^T | s_t=s]` under `q_k`.

Boundary:

```math
S_k(s,t) = \beta(s,t)\beta(s,t)^\top
\quad \text{if } s\in T \text{ or } t=H.
```

Backward recursion:

```math
S_k(s,t)
= \sum_{a,s'} q_k(a,s'\mid s,t)
  \Big(
    \psi\psi^\top
    + \psi\mu_k(s',t+1)^\top
    + \mu_k(s',t+1)\psi^\top
    + S_k(s',t+1)
  \Big),
```

where `psi = psi(s,a,s',t)`.

Then

```math
\Sigma_k(s,t) = S_k(s,t) - \mu_k(s,t)\mu_k(s,t)^\top.
```

This is the exact second-order local geometry at the anchor.

---

## 6.10 The reward atlas

A single anchor is not enough. We build a set of anchor rewards

```math
\mathcal A = \{r_1, r_2, \dots, r_K\}.
```

Each anchor stores:
- exact `h_k(s,t)`
- exact `A_k(s,t)`
- exact `mu_k(s,t)`
- exact `Sigma_k(s,t)`
- exact projected policy `pi_k_proj`
- diagnostic statistics and a calibrated trust radius

### Metric in reward space
The natural distance from target reward `r` to anchor `r_k` is not Euclidean.

Define a local metric from the start-state geometry:

```math
G_k = \Sigma_k(s_0,0) + \lambda I,
```

or an occupancy-weighted variant

```math
G_k = \sum_{t,s} \omega_k(s,t)\Sigma_k(s,t) + \lambda I.
```

Then define

```math
d_k(r)^2 = (r-r_k)^\top G_k (r-r_k).
```

This measures “how far the target reward is in path-measure geometry,” not just raw weight space.

---

## 6.11 Route-switch diagnostics

We need diagnostics that tell us when local geometry is becoming untrustworthy.

### 1. Curvature score

```math
C_k(r) = \delta^\top G_k \delta.
```

Large values mean the target reward pushes strongly along high-variance directions.

### 2. First-vs-second disagreement

```math
\Delta_{12,k}(r)
= \sum_{t,s} \omega_k(s,t)
  \left| \widehat A^{(2)}_{k,r}(s,t) - \widehat A^{(1)}_{k,r}(s,t) \right|.
```

If this is tiny, curvature correction is negligible. If large, first-order is likely underpowered.

### 3. Action margin
Let

```math
g^{(m)}_{k,r}(s,a,t)
= \log \pi_0(a\mid s,t) + \log \widehat Z^{(m)}_{k,r}(s,a,t),
```

and define the top-two action margin

```math
M_{k,r}(s,t) = g_{(1)} - g_{(2)}.
```

If this margin gets small at important states, the system is near a route-switch boundary.

### 4. Predicted action-flip rate
Compare the argmax action under the anchor policy and the target local approximation:

```math
F_k(r) = \sum_{t,s} \omega_k(s,t)
\mathbf 1\{ \arg\max_a \pi_k(a\mid s,t) \neq \arg\max_a \hat\pi_{k,r}(a\mid s,t) \}.
```

This is a concrete “route identity is changing” signal.

### 5. Realizability gap
Use `bar Gamma_r` and `Gamma_max_r` to tell whether stochasticity itself is making the target hard to realize.

---

## 6.12 Trust region and continuation

For each anchor `r_k`, calibrate a trust radius `rho_k` using held-out local perturbations.

A target reward is considered safe for direct use from anchor `k` if all are true:

```math
d_k(r) \le \rho_k,
\qquad
\Delta_{12,k}(r) \le \tau_\Delta,
\qquad
\min_{(s,t)\in\mathcal I_k} M_{k,r}(s,t) \ge \tau_M,
\qquad
\bar \Gamma_r \le \tau_\Gamma \; \text{(optional)}.
```

If not safe, do continuation.

### Continuation path
Starting from an anchor `r_cur`, move toward `r_tar` by a sequence

```math
r_0 = r_{cur},\; r_1,\; r_2,\; \dots,\; r_m = r_{tar},
```

with step sizes chosen so each hop stays inside trust.

Simplest rule:

```math
r_{j+1} = r_j + \alpha_j (r_{tar} - r_j),
```

where `alpha_j` is the largest value in `(0,1]` that keeps diagnostics below threshold.

### Two continuation modes
1. **Exact-anchor mode**
   At each intermediate `r_j`, solve `h`, `mu`, `Sigma` exactly and treat it as a new temporary anchor. This is the rigorous research benchmark.

2. **Approximate-hop mode**
   Use local geometry to hop without exact solving, then re-anchor only when needed. This is the scalable method later.

For v2, implement **exact-anchor mode first**. It gives the cleanest evidence.

---

## 7. Gold standards and methods to compare

For every target reward `r`, compare the following methods.

### Oracle objects
1. **Oracle tilt**: exact `q_r(a,s'|s,t)` from exact `h_r`
2. **Oracle projected policy**: exact `pi_r_proj` from exact `h_r`

### Methods
1. **Naive reuse**
   Reuse the base anchor policy with no geometry update.

2. **Single-anchor first-order**
   Use one fixed anchor, first-order `log h` approximation.

3. **Single-anchor second-order**
   Use one fixed anchor, second-order `log h` approximation.

4. **Atlas-direct first-order**
   Pick the best anchor, first-order local model.

5. **Atlas-direct second-order**
   Pick the best anchor, second-order local model.

6. **Atlas + continuation first-order**
   Safe multi-hop path through reward space.

7. **Atlas + continuation second-order**
   Safe multi-hop path with second-order charts.

8. **Exact solve baseline**
   Solve target reward exactly from scratch.

---

## 8. Metrics

## 8.1 Primary metrics

### Terminal KL
Compare terminal-state distributions induced by the approximate projected policy vs the oracle projected policy.

```math
D_{KL}(p^{\mathrm{oracle}}_{term} \| p^{\mathrm{approx}}_{term}).
```

### Occupancy KL
Compare state-time occupancy measures:

```math
D_{KL}(d^{\mathrm{oracle}} \| d^{\mathrm{approx}}).
```

This is mandatory. Terminal KL alone is too forgiving when many different policies end in the same place.

### Policy KL / TV
At each `(s,t)` compare approximate policy to oracle projected policy.

### Exact solve time vs transfer time
Measure wall-clock time of:
- full exact solve,
- single-anchor geometry,
- atlas selection,
- continuation.

---

## 8.2 Diagnostic metrics

- `realizability_gap_mean`
- `realizability_gap_max`
- `atlas_distance`
- `selected_anchor_id`
- `trust_region_hit` boolean
- `continuation_steps`
- `first_second_disagreement`
- `min_action_margin`
- `action_flip_rate`
- `geometry_rmse_logh`
- `geometry_relative_l2`
- `mean_episode_length`
- `path-risk mass` or environment-specific risk occupancy

---

## 8.3 Environment-specific semantic metrics

These are essential for interpretability.

### FrozenLake
- occupancy on states adjacent to holes
- occupancy on shortest-path corridor
- success probability
- average episode length

### CliffWalking
- occupancy on cells adjacent to the cliff
- occupancy on upper safe corridor
- success probability
- average episode length

### Taxi
- illegal-action rate
- pickup completion rate
- dropoff completion rate
- average episode length
- post-pickup wandering length

These semantic metrics are the story-tellers. They explain why KL changed.

---

## 9. Environment-specific feature maps

Keep reward dimension small and human-readable.

## 9.1 FrozenLake features
Suggested feature vector:

```text
[ goal_reached,
  hole_hit,
  step_cost ]
```

Optional later:
- `adjacent_to_hole` visitation bonus/penalty as an analysis-only feature

Reward examples:
- standard = `[+5, -5, -0.1]`
- goal_heavy = `[+10, -3, -0.1]`
- hole_penalty = `[+5, -10, -0.05]`
- fast_goal = `[+8, -5, -0.5]`
- cautious = `[+5, -8, -0.2]`
- explore = `[+3, -2, -0.01]`

## 9.2 CliffWalking features
Suggested feature vector:

```text
[ goal_reached,
  cliff_fall,
  step_cost ]
```

Reward examples:
- standard = `[+5, -10, -0.1]`
- goal_heavy = `[+10, -10, -0.1]`
- cliff_fear = `[+5, -20, -0.1]`
- fast_goal = `[+5, -10, -0.01]`
- cautious = `[+5, -15, -0.2]`
- explore = `[+3, -5, -0.01]`

## 9.3 Taxi features
Taxi needs extra care.

Suggested feature vector:

```text
[ successful_dropoff,
  illegal_action,
  step_cost,
  passenger_pickup_event ]
```

Critical requirement:
- **always use finite horizon `H`**
- avoid positive per-step shaping that rewards endless wandering

Reward examples:
- standard = `[+8, -5, -0.1, +1]`
- lawful = `[+8, -10, -0.1, +1]`
- rush = `[+8, -5, -0.5, +1]`
- pickup_biased = `[+5, -5, -0.1, +3]`
- cautious = `[+8, -8, -0.2, +1]`

Do **not** use an “explore forever” reward unless horizon is very explicit and results are interpreted as finite-horizon behavior.

---

## 10. Anchor set design

## 10.1 Initial anchors
Start with 6 to 12 anchors per environment.

Suggested anchor pool:
- the standard task
- each named task above
- 2 to 4 random tasks in the same bounded reward box

## 10.2 Adaptive anchor insertion
If a target reward produces:
- high terminal or occupancy KL,
- low action margin,
- large first-second disagreement,
- or requires many continuation steps,

then add it as a new anchor for future runs.

This turns the atlas into a growing memory of behavioral regimes.

---

## 11. Trust-region calibration

For each anchor `r_k`:
1. Sample small perturbations `r_k + epsilon` in random directions.
2. Solve exact target for each probe.
3. Compute direct first/second-order transfer error.
4. Fit a calibration curve from diagnostics to realized error.
5. Set trust radius `rho_k` as the largest geometric radius where error stays below threshold.

### Recommended thresholds
Start with:
- terminal KL < `1e-4`
- occupancy KL < `1e-3`
- min action margin > `0.1`
- first-second disagreement < `0.1`

Tune per environment.

---

## 12. Experiment plan

## Stage A. Reproduce current direct-transfer story
For each environment:
- exact solve for each task
- naive reuse
- single-anchor first-order
- single-anchor second-order
- report terminal KL, occupancy KL, geometry errors, realizability gap

Goal: reproduce the shape of the current plots under the new finite-horizon pipeline.

## Stage B. Add atlas-direct selection
For each target reward:
- choose anchor by geometric distance `d_k(r)`
- run direct first-order and second-order from that anchor
- compare against single-anchor methods

Goal: see whether “nearest local bowl” already fixes many failures.

## Stage C. Add continuation
For targets outside trust:
- perform safe multi-hop continuation from chosen anchor
- compare direct vs continuation
- log number of steps and intermediate anchors used

Goal: show that route-switch boundaries are better crossed by several small tilts than one large jump.

## Stage D. Failure analysis
Focus on rewards where direct second-order is worse than first-order or worse than naive.

For each such case, log:
- selected anchor
- atlas distance
- action margin collapse
- first-second disagreement
- realizability gap
- continuation result
- semantic occupancy changes

Goal: turn failures into interpretable case studies.

## Stage E. Taxi stabilization
Run Taxi with several horizons, e.g. `H ∈ {20, 40, 80}`.

Measure:
- stability of exact solves
- stability of geometry
- terminal and occupancy KL
- mean episode length
- whether earlier pathologies disappear

Goal: confirm that the loop issue was a horizon/path-mass problem, not a death sentence for the method.

---

## 13. Result outputs

Each experiment should write:

### JSON summary per environment

```json
{
  "env": "FrozenLake-v1",
  "variant": "slippery_8x8",
  "horizon": 100,
  "anchors": [...],
  "tasks": [
    {
      "task": "hole_penalty",
      "reward": [5, -10, -0.05],
      "selected_anchor": "standard",
      "atlas_distance": 2.31,
      "continuation_steps": 3,
      "terminal_kl": {
        "naive": 1.8e-4,
        "single_order1": 2.6e-6,
        "single_order2": 1.4e-7,
        "atlas_order2": 9.0e-8,
        "cont_order2": 2.0e-8
      },
      "occupancy_kl": {...},
      "policy_kl": {...},
      "realizability_gap_mean": 0.012,
      "realizability_gap_max": 0.081,
      "first_second_disagreement": 0.21,
      "min_action_margin": 0.04,
      "action_flip_rate": 0.17,
      "semantic_metrics": {
        "hole_adjacent_occupancy": 0.11,
        "success_prob": 0.88,
        "mean_episode_length": 23.7
      }
    }
  ]
}
```

### PNG plots
Required plots:
1. terminal KL vs reward distance
2. occupancy KL vs reward distance
3. direct vs continuation comparison
4. predicted geometry error vs realized KL scatter
5. realizability gap vs stochasticity / environment variant
6. action margin vs failure probability
7. semantic occupancy heatmaps for selected case studies
8. anchor atlas visualization in 2D reward slices

### Markdown report
One report per environment plus one combined summary.

---

## 14. Plot design for interpretability

Every plot must answer one question.

### Plot A. `terminal_kl_vs_distance.png`
Question: does local transfer degrade with reward distance?

### Plot B. `occupancy_kl_vs_distance.png`
Question: are we matching not just the ending, but the route distribution?

### Plot C. `direct_vs_continuation.png`
Question: does continuation rescue large or boundary-crossing shifts?

### Plot D. `predicted_vs_realized_error.png`
Question: do our diagnostics actually predict failure?

### Plot E. `margin_vs_error.png`
Question: are route-switch boundaries visible through action-margin collapse?

### Plot F. `realizability_gap_vs_error.png`
Question: how much of the problem is due to stochastic non-realizability?

### Plot G. `semantic_case_study_<task>.png`
Question: what behavioral regime changed?

For FrozenLake and CliffWalking, include occupancy heatmaps and difference heatmaps.

---

## 15. Repo structure

```text
reward_atlas_transfer/
  README.md
  pyproject.toml
  requirements.txt
  src/
    envs/
      build_tabular_model.py
      frozenlake_features.py
      cliffwalking_features.py
      taxi_features.py
    core/
      dp.py
      bridge_projection.py
      geometry.py
      atlas.py
      continuation.py
      metrics.py
      diagnostics.py
      occupancy.py
    experiments/
      run_direct_transfer.py
      run_atlas_transfer.py
      run_continuation_transfer.py
      run_taxi_horizon_sweep.py
      plot_results.py
    utils/
      io.py
      seeding.py
      logging.py
  configs/
    frozenlake_4x4_det.yaml
    frozenlake_4x4_slip.yaml
    frozenlake_8x8_det.yaml
    frozenlake_8x8_slip.yaml
    cliffwalking.yaml
    taxi_h20.yaml
    taxi_h40.yaml
    taxi_h80.yaml
  outputs/
    ...
```

---

## 16. Python implementation skeleton

## 16.1 Core dataclasses

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TabularEpisodicModel:
    nS: int
    nA: int
    H: int
    P: np.ndarray          # shape [nS, nA, nS]
    terminal: np.ndarray   # shape [nS], bool
    rho0: np.ndarray       # shape [nS]

@dataclass
class RewardSpec:
    dim: int
    step_features: np.ndarray      # shape [H, nS, nA, nS, d] or [nS, nA, nS, d]
    terminal_features: np.ndarray  # shape [H+1, nS, d] or [nS, d]

@dataclass
class Anchor:
    name: str
    r: np.ndarray                  # [d]
    log_h: np.ndarray              # [H+1, nS]
    h: np.ndarray                  # [H+1, nS]
    mu: np.ndarray                 # [H+1, nS, d]
    Sigma: np.ndarray              # [H+1, nS, d, d]
    policy: np.ndarray             # [H, nS, nA]
    metric: np.ndarray             # [d, d]
    trust_radius: float
    diagnostics: dict
```

## 16.2 Exact backward solve

```python
import numpy as np


def solve_h(model: TabularEpisodicModel,
            reward: RewardSpec,
            pi0: np.ndarray,
            r: np.ndarray) -> np.ndarray:
    H, nS, nA = model.H, model.nS, model.nA
    h = np.zeros((H + 1, nS), dtype=np.float64)

    # boundary
    for s in range(nS):
        h[H, s] = np.exp(reward.terminal_features[H, s] @ r)
    for t in range(H - 1, -1, -1):
        for s in range(nS):
            if model.terminal[s]:
                h[t, s] = np.exp(reward.terminal_features[t, s] @ r)
                continue
            total = 0.0
            for a in range(nA):
                z = 0.0
                for sp in range(nS):
                    p = model.P[s, a, sp]
                    if p == 0.0:
                        continue
                    feat = reward.step_features[t, s, a, sp]
                    z += p * np.exp(feat @ r) * h[t + 1, sp]
                total += pi0[t, s, a] * z
            h[t, s] = total
    return h
```

## 16.3 Exact bridge projection

```python

def project_policy(model, reward, pi0, r, h):
    H, nS, nA = model.H, model.nS, model.nA
    pi = np.zeros((H, nS, nA), dtype=np.float64)
    z_sa = np.zeros((H, nS, nA), dtype=np.float64)

    for t in range(H):
        for s in range(nS):
            if model.terminal[s]:
                continue
            denom = h[t, s]
            if denom <= 0:
                continue
            for a in range(nA):
                z = 0.0
                for sp in range(nS):
                    p = model.P[s, a, sp]
                    if p == 0.0:
                        continue
                    feat = reward.step_features[t, s, a, sp]
                    z += p * np.exp(feat @ r) * h[t + 1, sp]
                z_sa[t, s, a] = z
            numer = pi0[t, s] * z_sa[t, s]
            total = numer.sum()
            if total > 0:
                pi[t, s] = numer / total
            else:
                pi[t, s] = 1.0 / nA
    return pi, z_sa
```

## 16.4 Exact moment solve

```python

def solve_moments(model, reward, pi0, r, h):
    H, nS, nA = model.H, model.nS, model.nA
    d = reward.terminal_features.shape[-1]
    mu = np.zeros((H + 1, nS, d), dtype=np.float64)
    S = np.zeros((H + 1, nS, d, d), dtype=np.float64)

    for t in range(H, -1, -1):
        for s in range(nS):
            if model.terminal[s] or t == H:
                b = reward.terminal_features[t, s]
                mu[t, s] = b
                S[t, s] = np.outer(b, b)

    for t in range(H - 1, -1, -1):
        for s in range(nS):
            if model.terminal[s]:
                continue
            for a in range(nA):
                for sp in range(nS):
                    p = model.P[s, a, sp]
                    if p == 0.0:
                        continue
                    feat = reward.step_features[t, s, a, sp]
                    q = pi0[t, s, a] * p * np.exp(feat @ r) * h[t + 1, sp] / h[t, s]

                    # Phi = feat + Phi_next
                    mu_next = mu[t + 1, sp]
                    S_next = S[t + 1, sp]
                    mu[t, s] += q * (feat + mu_next)
                    S[t, s] += q * (
                        np.outer(feat, feat)
                        + np.outer(feat, mu_next)
                        + np.outer(mu_next, feat)
                        + S_next
                    )

    Sigma = S - np.einsum('tsi,tsj->tsij', mu, mu)
    return mu, Sigma
```

This is the exact uncentered-second-moment recursion corresponding to the formula in Section 6.9.

## 16.5 Local approximation from an anchor

```python

def approx_log_h(anchor: Anchor, r_target: np.ndarray, order: int) -> np.ndarray:
    delta = r_target - anchor.r
    log_h = anchor.log_h + np.tensordot(anchor.mu, delta, axes=([2], [0]))
    if order >= 2:
        quad = 0.5 * np.einsum('i,tsij,j->ts', delta, anchor.Sigma, delta)
        log_h = log_h + quad
    return log_h
```

## 16.6 Anchor selection

```python

def anchor_distance(anchor: Anchor, r_target: np.ndarray, lam: float = 1e-8) -> float:
    delta = r_target - anchor.r
    G = anchor.metric + lam * np.eye(anchor.metric.shape[0])
    return float(np.sqrt(delta @ G @ delta))


def select_anchor(anchors, r_target):
    scored = [(anchor_distance(a, r_target), a) for a in anchors]
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored
```

## 16.7 Continuation

```python

def continuation_path(r_start, r_target, propose_ok):
    pts = [r_start.copy()]
    r_cur = r_start.copy()
    while np.linalg.norm(r_target - r_cur) > 1e-12:
        alpha_lo, alpha_hi = 0.0, 1.0
        # backtracking / binary search for largest safe hop
        for _ in range(24):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            r_next = r_cur + alpha * (r_target - r_cur)
            if propose_ok(r_cur, r_next):
                alpha_lo = alpha
            else:
                alpha_hi = alpha
        alpha = max(alpha_lo, 1e-3)
        r_cur = r_cur + alpha * (r_target - r_cur)
        pts.append(r_cur.copy())
        if len(pts) > 64:
            break
    pts[-1] = r_target.copy()
    return pts
```

---

## 17. Evaluation protocol

### For each environment variant
1. Build exact tabular model and feature map.
2. Choose horizon `H`.
3. Choose base reference policy `pi0`.
4. Build anchor pool and exact anchor objects.
5. Sample target rewards:
   - named tasks
   - random rewards inside a bounded box
   - dense 2D slices for visualization
6. For each target reward:
   - exact oracle solve
   - run all methods
   - compute all metrics and diagnostics
7. Save JSON, plots, and markdown summary.

### Random reward sampling
For each env, sample 100 to 500 random target rewards from a bounded box around the task family. Use these for scatter plots and calibration, not just named tasks.

### 2D reward slices
For interpretability, freeze all but two coordinates and scan a grid.

Examples:
- FrozenLake: vary `(goal_weight, hole_weight)` with fixed step cost
- CliffWalking: vary `(cliff_penalty, step_cost)` with fixed goal reward
- Taxi: vary `(illegal_penalty, step_cost)` with fixed success/pickup reward

This is how you draw the atlas and the route-switch frontier.

---

## 18. Acceptance criteria

### Must-have
- deterministic FrozenLake and CliffWalking: atlas and continuation do not regress relative to current direct geometry
- slippery FrozenLake: continuation improves over direct second-order on difficult tasks
- occupancy KL is reported everywhere, not just terminal KL
- Taxi becomes numerically stable under finite horizon
- diagnostics correlate with realized failure

### Strong success
- atlas-direct beats single-anchor direct on average
- continuation beats atlas-direct near reward boundaries
- action-margin collapse and first-second disagreement predict failure with useful precision
- semantic metrics explain the KL changes in case studies

---

## 19. Expected qualitative outcomes

If the theory is right, results should look like this:

1. **Small shifts**
   second-order direct transfer is best and very fast.

2. **Moderate shifts within same behavioral regime**
   atlas-direct should beat single-anchor methods.

3. **Shifts near route-switch boundaries**
   direct second-order may wobble; continuation should help.

4. **Deterministic environments**
   realizability gap is near zero.

5. **Stochastic environments**
   realizability gap is nonzero but bridge projection keeps transfer useful.

6. **Taxi without horizon**
   unstable or pathological.

7. **Taxi with horizon**
   stable enough for interpretable transfer experiments.

---

## 20. Risks and how to de-risk them

### Risk 1. Second-order still fails often
Response:
- tighten trust radii
- use continuation more aggressively
- add more anchors

### Risk 2. Too many anchors, too little interpretability
Response:
- keep 2D reward slices
- cluster anchors by semantic behavior
- report occupancy heatmaps for representative anchors only

### Risk 3. Taxi still behaves strangely
Response:
- shorten horizon
- simplify reward family
- use stronger negative step cost
- inspect mean episode length and looping occupancy

### Risk 4. Metrics are noisy or too many
Response:
- promote a small dashboard:
  - terminal KL
  - occupancy KL
  - realizability gap mean
  - first-second disagreement
  - min action margin
  - continuation steps

---

## 21. Concrete milestone plan

### Milestone 1. Stable exact pipeline
- build exact time-augmented solver
- verify on FrozenLake and CliffWalking
- verify Taxi finite-horizon stability

### Milestone 2. Reward atlas
- implement anchors
- implement anchor distance metric
- implement atlas-direct transfer

### Milestone 3. Diagnostics
- compute curvature, margin, disagreement, realizability gap
- produce scatter plots vs realized error

### Milestone 4. Continuation
- implement safe-step search
- exact-anchor continuation
- direct vs continuation benchmarks

### Milestone 5. Reports
- environment-specific markdown reports
- combined summary tables and plots
- semantic case studies

---

## 22. One-sentence research story

> **We model reward transfer not as learning a new policy, but as navigating a statistical atlas of reward-tilted future clouds, projecting each tilt back into policy space, and switching local charts when behavioral geometry bends too sharply.**

