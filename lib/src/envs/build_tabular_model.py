"""Build explicit tabular episodic models from Gymnasium environments.

All research pipelines operate on the TabularEpisodicModel and RewardSpec
dataclasses, never on raw Gymnasium wrappers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym


@dataclass
class TabularEpisodicModel:
    """Time-augmented finite-horizon tabular MDP.

    All solvers index by (t, s) where t ∈ {0, ..., H}.
    """
    nS: int
    nA: int
    H: int
    P: np.ndarray           # [nS, nA, nS]
    terminal: np.ndarray    # [nS] bool
    rho0: np.ndarray        # [nS]
    legal: np.ndarray       # [nS, nA] bool

    def nonterminal_mask(self) -> np.ndarray:
        return ~self.terminal

    def nonterminal_indices(self) -> np.ndarray:
        return np.where(~self.terminal)[0]

    def terminal_indices(self) -> np.ndarray:
        return np.where(self.terminal)[0]

    def validate(self) -> None:
        assert self.P.shape == (self.nS, self.nA, self.nS)
        assert self.terminal.shape == (self.nS,)
        assert self.rho0.shape == (self.nS,)
        assert self.legal.shape == (self.nS, self.nA)
        active = self.legal & (~self.terminal[:, None])
        row_sums = self.P.sum(axis=2)
        if active.any():
            max_err = np.abs(row_sums[active] - 1.0).max()
            assert max_err < 1e-10, f"Transition rows off by {max_err}"


@dataclass
class RewardSpec:
    """Feature-based reward specification for an episodic model.

    step_features: [H, nS, nA, nS, d] or [nS, nA, nS, d] (broadcast over time)
    terminal_features: [H+1, nS, d] or [nS, d] (broadcast over time)
    """
    dim: int
    step_features: np.ndarray
    terminal_features: np.ndarray
    time_varying: bool = False

    def get_step_feat(self, t: int) -> np.ndarray:
        """Get step features for time t. Shape [nS, nA, nS, d]."""
        if self.step_features.ndim == 5:
            return self.step_features[t]
        return self.step_features

    def get_term_feat(self, t: int) -> np.ndarray:
        """Get terminal features for time t. Shape [nS, d]."""
        if self.terminal_features.ndim == 3:
            return self.terminal_features[t]
        return self.terminal_features

    def expand_to_time(self, H: int) -> RewardSpec:
        """Return a copy with explicit time-expanded arrays."""
        if self.step_features.ndim == 5:
            sf = self.step_features
        else:
            sf = np.broadcast_to(
                self.step_features[None, ...],
                (H,) + self.step_features.shape
            ).copy()
        if self.terminal_features.ndim == 3:
            tf = self.terminal_features
        else:
            tf = np.broadcast_to(
                self.terminal_features[None, ...],
                (H + 1,) + self.terminal_features.shape
            ).copy()
        return RewardSpec(dim=self.dim, step_features=sf,
                         terminal_features=tf, time_varying=True)


@dataclass
class Anchor:
    """An anchor point in reward space with precomputed local geometry."""
    name: str
    r: np.ndarray               # [d]
    h: np.ndarray               # [H+1, nS]
    log_h: np.ndarray           # [H+1, nS]
    mu: np.ndarray              # [H+1, nS, d]
    Sigma: np.ndarray           # [H+1, nS, d, d]
    policy: np.ndarray          # [H, nS, nA]
    metric: np.ndarray          # [d, d]
    trust_radius: float = 1.0
    diagnostics: dict = field(default_factory=dict)


def extract_tabular_model(env_name: str, H: int, **env_kwargs) -> TabularEpisodicModel:
    """Extract a TabularEpisodicModel from a Gymnasium Toy Text environment."""
    env = gym.make(env_name, **env_kwargs)
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise ValueError(f"{env_name} does not expose a tabular .P attribute")

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
            for prob, s_next, _reward, done in outcomes:
                P[s, a, s_next] += prob
                if done:
                    terminal[s_next] = True

    rho0 = np.zeros(nS, dtype=np.float64)
    if hasattr(base, "initial_state_distrib"):
        rho0[:] = np.asarray(base.initial_state_distrib, dtype=np.float64)
    else:
        obs, _ = env.reset(seed=0)
        rho0[int(obs)] = 1.0

    env.close()
    return TabularEpisodicModel(nS=nS, nA=nA, H=H, P=P, terminal=terminal,
                                 rho0=rho0, legal=legal)


def get_env_desc(env_name: str, **env_kwargs) -> np.ndarray | None:
    """Get the map description for grid-based environments."""
    env = gym.make(env_name, **env_kwargs)
    base = env.unwrapped
    desc = np.asarray(base.desc) if hasattr(base, "desc") else None
    env.close()
    return desc


def uniform_policy(model: TabularEpisodicModel) -> np.ndarray:
    """Uniform reference policy pi0[t, s, a], shape [H, nS, nA].

    Time-independent but stored with time axis for consistency.
    """
    pi_slice = model.legal.astype(np.float64)
    row_sums = pi_slice.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    pi_slice = pi_slice / row_sums
    return np.broadcast_to(pi_slice[None, ...], (model.H, model.nS, model.nA)).copy()
