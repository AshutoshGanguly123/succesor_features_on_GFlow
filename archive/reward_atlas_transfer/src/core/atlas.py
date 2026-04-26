"""Reward atlas: a collection of anchors with geometric distance and selection.

The atlas stores precomputed local geometry at multiple anchor rewards.
Selection uses the natural Riemannian-like distance d_k(r) from Section 6.10.
"""

import numpy as np
from src.envs.build_tabular_model import (
    TabularEpisodicModel, RewardSpec, Anchor
)
from src.core.dp import solve_h
from src.core.bridge_projection import project_policy
from src.core.geometry import solve_moments, approx_h


def anchor_distance(anchor: Anchor, r_target: np.ndarray,
                    lam: float = 1e-6) -> float:
    """Geometric distance from anchor to target reward.

    d_k(r)^2 = (r - r_k)^T G_k (r - r_k)
    where G_k = Sigma_k(s0, 0) + lambda * I (or occupancy-weighted).
    """
    delta = r_target - anchor.r
    G = anchor.metric + lam * np.eye(anchor.metric.shape[0])
    return float(np.sqrt(max(0, delta @ G @ delta)))


def euclidean_distance(anchor: Anchor, r_target: np.ndarray) -> float:
    return float(np.linalg.norm(r_target - anchor.r))


def select_anchor(anchors: list[Anchor], r_target: np.ndarray,
                  use_geometric: bool = True) -> tuple[Anchor, list[tuple[float, Anchor]]]:
    """Select the best anchor for a target reward.

    Returns:
        best_anchor: The selected anchor.
        scored: List of (distance, anchor) sorted by distance.
    """
    dist_fn = anchor_distance if use_geometric else euclidean_distance
    scored = [(dist_fn(a, r_target), a) for a in anchors]
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored


def build_anchor(name: str,
                 r: np.ndarray,
                 model: TabularEpisodicModel,
                 reward: RewardSpec,
                 pi0: np.ndarray,
                 trust_radius: float = 1.0) -> Anchor:
    """Build a complete anchor with all precomputed quantities."""
    h = solve_h(model, reward, pi0, r)
    log_h = np.log(np.maximum(h, 1e-300))
    mu, Sigma = solve_moments(model, reward, pi0, r, h)
    pi, _ = project_policy(model, reward, pi0, r, h)

    # Metric: occupancy-weighted average covariance or start-state covariance
    metric = Sigma[0, 0] if not model.terminal[0] else np.eye(reward.dim)

    return Anchor(
        name=name, r=r.copy(), h=h, log_h=log_h,
        mu=mu, Sigma=Sigma, policy=pi,
        metric=metric, trust_radius=trust_radius,
    )


def build_atlas(anchor_specs: list[dict],
                model: TabularEpisodicModel,
                reward: RewardSpec,
                pi0: np.ndarray,
                trust_radius: float = 1.0) -> list[Anchor]:
    """Build a full atlas from a list of anchor specifications.

    Each spec is a dict with 'name' and 'r' keys.
    """
    anchors = []
    for spec in anchor_specs:
        r = np.array(spec["r"], dtype=np.float64)
        anchor = build_anchor(spec["name"], r, model, reward, pi0, trust_radius)
        anchors.append(anchor)
    return anchors


def transfer_from_anchor(anchor: Anchor,
                         r_target: np.ndarray,
                         model: TabularEpisodicModel,
                         reward: RewardSpec,
                         pi0: np.ndarray,
                         order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Perform local geometry transfer from an anchor.

    Returns:
        pi_approx: shape [H, nS, nA], approximate projected policy.
        h_approx: shape [H+1, nS], approximate successor mass.
    """
    h_approx = approx_h(anchor, r_target, order=order)
    pi_approx, _ = project_policy(model, reward, pi0, r_target, h_approx)
    return pi_approx, h_approx
