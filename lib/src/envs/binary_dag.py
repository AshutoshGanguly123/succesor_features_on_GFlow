"""Deterministic complete binary DAG utilities for report Q1-Q3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BinaryDAG:
    """Complete binary tree viewed as a DAG."""

    depth: int
    terminal_features: np.ndarray  # [n_terminals, d]

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.terminal_features.ndim != 2:
            raise ValueError("terminal_features must be [n_terminals, d]")
        expected = 2**self.depth
        if self.terminal_features.shape[0] != expected:
            raise ValueError(
                f"expected {expected} terminal rows, got {self.terminal_features.shape[0]}"
            )

        self.n_internal = 2**self.depth - 1
        self.n_terminals = 2**self.depth
        self.n_nodes = self.n_internal + self.n_terminals
        self.root = 0
        self.terminal_offset = self.n_internal
        self.feature_dim = self.terminal_features.shape[1]

    @classmethod
    def from_random(
        cls,
        depth: int = 8,
        feature_dim: int = 4,
        seed: int = 0,
        feature_scale: float = 1.0,
    ) -> "BinaryDAG":
        rng = np.random.default_rng(seed)
        feats = rng.normal(0.0, feature_scale, size=(2**depth, feature_dim))
        return cls(depth=depth, terminal_features=feats.astype(np.float64))

    @property
    def terminal_indices(self) -> np.ndarray:
        return np.arange(self.terminal_offset, self.n_nodes, dtype=np.int64)

    def is_terminal(self, node: int) -> bool:
        return node >= self.terminal_offset

    def children(self, node: int) -> tuple[int, int]:
        if self.is_terminal(node):
            raise ValueError("terminal nodes do not have children")
        return 2 * node + 1, 2 * node + 2

    def leaf_rewards(self, w: np.ndarray, reward_family: str) -> np.ndarray:
        dot = self.terminal_features @ w
        if reward_family == "linear":
            return dot
        if reward_family == "loglinear":
            dot = np.clip(dot, -500, 500)
            return np.exp(dot)
        raise ValueError("reward_family must be 'linear' or 'loglinear'")

    def reference_flow_and_policy(self, w0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Optimal flow and forward policy at reference reward w0."""
        rewards = self.leaf_rewards(w0, reward_family="loglinear")
        flow = np.zeros(self.n_nodes, dtype=np.float64)
        flow[self.terminal_offset :] = rewards

        for node in range(self.n_internal - 1, -1, -1):
            lch, rch = self.children(node)
            flow[node] = flow[lch] + flow[rch]

        policy = np.zeros((self.n_internal, 2), dtype=np.float64)
        for node in range(self.n_internal):
            lch, rch = self.children(node)
            denom = max(flow[node], 1e-300)
            policy[node, 0] = flow[lch] / denom
            policy[node, 1] = flow[rch] / denom
        return flow, policy

    def descendant_measure(self, policy: np.ndarray) -> np.ndarray:
        """mu[node, leaf] under the given internal forward policy."""
        if policy.shape != (self.n_internal, 2):
            raise ValueError("policy must have shape [n_internal, 2]")

        mu = np.zeros((self.n_nodes, self.n_terminals), dtype=np.float64)
        for i, node in enumerate(self.terminal_indices):
            mu[node, i] = 1.0

        for node in range(self.n_internal - 1, -1, -1):
            lch, rch = self.children(node)
            mu[node] = policy[node, 0] * mu[lch] + policy[node, 1] * mu[rch]
        return mu

    def successor_features(self, mu: np.ndarray) -> np.ndarray:
        """Psi(node) = E_mu[phi]."""
        return mu @ self.terminal_features

    def h_from_mu(self, mu: np.ndarray, w: np.ndarray, reward_family: str) -> np.ndarray:
        return mu @ self.leaf_rewards(w, reward_family=reward_family)

    def log_h_from_mu(self, mu: np.ndarray, w: np.ndarray, reward_family: str) -> np.ndarray:
        h = self.h_from_mu(mu, w, reward_family)
        return np.log(np.maximum(h, 1e-300))

    def induced_policy_from_h(self, h: np.ndarray, ref_policy: np.ndarray) -> np.ndarray:
        """Construct forward policy from h and reference policy, no step reward."""
        if h.shape[0] != self.n_nodes:
            raise ValueError("h has wrong shape")
        if ref_policy.shape != (self.n_internal, 2):
            raise ValueError("ref_policy has wrong shape")

        policy = np.zeros_like(ref_policy)
        for node in range(self.n_internal):
            lch, rch = self.children(node)
            left = ref_policy[node, 0] * h[lch]
            right = ref_policy[node, 1] * h[rch]
            denom = max(left + right, 1e-300)
            policy[node, 0] = left / denom
            policy[node, 1] = right / denom
        return policy

    def terminal_distribution_from_policy(self, policy: np.ndarray) -> np.ndarray:
        """Terminal distribution reached from root under internal policy."""
        probs = np.zeros(self.n_nodes, dtype=np.float64)
        probs[self.root] = 1.0
        for node in range(self.n_internal):
            lch, rch = self.children(node)
            mass = probs[node]
            probs[lch] += mass * policy[node, 0]
            probs[rch] += mass * policy[node, 1]
        out = probs[self.terminal_offset :]
        out /= max(out.sum(), 1e-300)
        return out


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))
