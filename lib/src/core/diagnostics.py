"""Route-switch diagnostics from PRD Section 6.11.

Curvature score, first-vs-second disagreement, action margin,
action-flip rate, realizability gap.
"""

import numpy as np
from src.envs.build_tabular_model import TabularEpisodicModel, RewardSpec, Anchor
from src.core.geometry import approx_log_h
from src.core.bridge_projection import (
    compute_action_scores, project_policy, tilted_joint_law
)


def curvature_score(anchor: Anchor, r_target: np.ndarray) -> float:
    """C_k(r) = delta^T G_k delta. Large = high-variance direction."""
    delta = r_target - anchor.r
    lam = 1e-6
    G = anchor.metric + lam * np.eye(len(delta))
    return float(delta @ G @ delta)


def first_second_disagreement(anchor: Anchor,
                              r_target: np.ndarray,
                              model: TabularEpisodicModel,
                              occupancy_weights: np.ndarray = None) -> float:
    """Delta_12 = weighted sum of |A^(2) - A^(1)| over (t,s).

    Measures how much the quadratic term changes the approximation.
    """
    log_h_o1 = approx_log_h(anchor, r_target, order=1)
    log_h_o2 = approx_log_h(anchor, r_target, order=2)
    diff = np.abs(log_h_o2 - log_h_o1)

    if occupancy_weights is not None:
        return float(np.sum(occupancy_weights * diff))

    nt_mask = ~model.terminal
    # Average over nonterminal states, all time steps
    return float(diff[:, nt_mask].mean()) if nt_mask.any() else 0.0


def action_margin(model: TabularEpisodicModel,
                  pi: np.ndarray) -> np.ndarray:
    """Top-two action margin at each (t, s).

    M(s,t) = log_score(best action) - log_score(second best action).

    Returns:
        margins: shape [H, nS].
    """
    H, nS, nA = pi.shape
    margins = np.zeros((H, nS), dtype=np.float64)

    for t in range(H):
        for s in range(nS):
            if model.terminal[s]:
                continue
            log_p = np.log(np.maximum(pi[t, s], 1e-300))
            sorted_lp = np.sort(log_p)[::-1]
            if nA >= 2:
                margins[t, s] = sorted_lp[0] - sorted_lp[1]
            else:
                margins[t, s] = float("inf")

    return margins


def min_action_margin(model: TabularEpisodicModel,
                      pi: np.ndarray) -> float:
    """Minimum action margin over nonterminal (t,s) pairs."""
    margins = action_margin(model, pi)
    nt_vals = []
    for t in range(model.H):
        for s in model.nonterminal_indices():
            nt_vals.append(margins[t, s])
    return float(min(nt_vals)) if nt_vals else float("inf")


def action_flip_rate(model: TabularEpisodicModel,
                     pi_anchor: np.ndarray,
                     pi_target: np.ndarray,
                     occupancy_weights: np.ndarray = None) -> float:
    """Fraction of (t,s) where the argmax action changes.

    F_k(r) = sum_{t,s} w(s,t) * 1[argmax pi_k != argmax pi_target]
    """
    H = pi_anchor.shape[0]
    flips = 0.0
    total = 0.0

    for t in range(H):
        for s in model.nonterminal_indices():
            w = 1.0 if occupancy_weights is None else occupancy_weights[t, s]
            if pi_anchor[t, s].argmax() != pi_target[t, s].argmax():
                flips += w
            total += w

    return float(flips / max(total, 1e-10))


def realizability_gap(model: TabularEpisodicModel,
                      q_tilted: np.ndarray,
                      pi_proj: np.ndarray,
                      eps: float = 1e-30) -> np.ndarray:
    """Per-(t,s) realizability gap: KL(q(a,s'|s,t) || pi(a|s,t) P(s'|s,a)).

    Returns:
        gap: shape [H, nS].
    """
    H, nS, nA = pi_proj.shape
    gap = np.zeros((H, nS), dtype=np.float64)

    for t in range(H):
        realizable = pi_proj[t, :, :, None] * model.P  # [nS, nA, nS]
        for s in model.nonterminal_indices():
            q_flat = q_tilted[t, s].ravel()
            r_flat = realizable[s].ravel()
            mask = q_flat > eps
            if not mask.any():
                continue
            gap[t, s] = np.sum(
                q_flat[mask] * np.log(q_flat[mask] / np.maximum(r_flat[mask], eps))
            )

    return gap


def compute_all_diagnostics(model: TabularEpisodicModel,
                            reward: RewardSpec,
                            pi0: np.ndarray,
                            r_target: np.ndarray,
                            h_target: np.ndarray,
                            pi_target: np.ndarray,
                            anchor: Anchor) -> dict:
    """Compute all diagnostics for a target reward relative to an anchor."""
    diags = {}
    diags["curvature_score"] = curvature_score(anchor, r_target)
    diags["first_second_disagreement"] = first_second_disagreement(
        anchor, r_target, model
    )
    diags["min_action_margin"] = min_action_margin(model, pi_target)
    diags["action_flip_rate"] = action_flip_rate(
        model, anchor.policy, pi_target
    )

    q = tilted_joint_law(model, reward, pi0, r_target, h_target)
    gap = realizability_gap(model, q, pi_target)
    nt = model.nonterminal_mask()
    diags["realizability_gap_mean"] = float(gap[:, nt].mean()) if nt.any() else 0.0
    diags["realizability_gap_max"] = float(gap[:, nt].max()) if nt.any() else 0.0
    diags["atlas_distance"] = float(np.linalg.norm(r_target - anchor.r))

    return diags
