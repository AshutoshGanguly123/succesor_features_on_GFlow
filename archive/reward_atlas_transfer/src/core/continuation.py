"""Continuation / homotopy through reward space.

When a target reward is outside the trust region of any anchor,
we walk from the nearest anchor toward the target in small safe steps.
Each step re-solves exactly (exact-anchor mode) for rigorous benchmarking.
"""

import numpy as np
from src.envs.build_tabular_model import (
    TabularEpisodicModel, RewardSpec, Anchor
)
from src.core.atlas import build_anchor, anchor_distance
from src.core.diagnostics import compute_all_diagnostics


def _is_safe(diagnostics: dict, thresholds: dict) -> bool:
    """Check if diagnostics are within safe thresholds."""
    if diagnostics.get("first_second_disagreement", 0) > thresholds.get("max_disagreement", 0.5):
        return False
    if diagnostics.get("min_action_margin", 1.0) < thresholds.get("min_margin", 0.05):
        return False
    return True


def continuation_path(r_start: np.ndarray,
                      r_target: np.ndarray,
                      model: TabularEpisodicModel,
                      reward: RewardSpec,
                      pi0: np.ndarray,
                      thresholds: dict = None,
                      max_steps: int = 32) -> list[Anchor]:
    """Walk from r_start to r_target via safe intermediate anchors.

    Uses exact-anchor mode: each intermediate point is solved exactly
    and becomes a temporary anchor for the next step.

    Args:
        r_start: Starting reward vector.
        r_target: Target reward vector.
        model: Tabular episodic model.
        reward: Reward specification.
        pi0: Reference policy.
        thresholds: Dict of safety thresholds.
        max_steps: Maximum continuation hops.

    Returns:
        path: List of Anchor objects along the path (including start and end).
    """
    if thresholds is None:
        thresholds = {
            "max_disagreement": 0.5,
            "min_margin": 0.05,
        }

    current = build_anchor("cont_start", r_start, model, reward, pi0)
    path = [current]

    for step in range(max_steps):
        remaining = r_target - current.r
        if np.linalg.norm(remaining) < 1e-10:
            break

        # Binary search for the largest safe alpha
        alpha_lo, alpha_hi = 0.0, 1.0
        best_alpha = 0.0

        for _ in range(20):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            r_candidate = current.r + alpha * remaining

            candidate = build_anchor(
                f"cont_step_{step}_{alpha:.3f}",
                r_candidate, model, reward, pi0
            )
            diags = compute_all_diagnostics(
                model, reward, pi0, r_candidate,
                candidate.h, candidate.policy, current
            )

            if _is_safe(diags, thresholds):
                alpha_lo = alpha
                best_alpha = alpha
            else:
                alpha_hi = alpha

        # Take the step
        alpha = max(best_alpha, 0.05)
        if alpha >= 0.99:
            alpha = 1.0

        r_next = current.r + alpha * remaining
        next_anchor = build_anchor(
            f"cont_{step}", r_next, model, reward, pi0
        )
        path.append(next_anchor)
        current = next_anchor

        if np.linalg.norm(r_target - current.r) < 1e-10:
            break

    # Final exact solve at target
    if np.linalg.norm(r_target - current.r) > 1e-10:
        final = build_anchor("cont_final", r_target, model, reward, pi0)
        path.append(final)

    return path


def continuation_transfer(start_anchor: Anchor,
                          r_target: np.ndarray,
                          model: TabularEpisodicModel,
                          reward: RewardSpec,
                          pi0: np.ndarray,
                          thresholds: dict = None) -> dict:
    """Run continuation and return the final policy + metadata.

    Returns dict with keys: policy, h, path_length, anchors_used.
    """
    path = continuation_path(
        start_anchor.r, r_target, model, reward, pi0, thresholds
    )
    final = path[-1]
    return {
        "policy": final.policy,
        "h": final.h,
        "path_length": len(path),
        "anchors_used": [a.name for a in path],
        "final_anchor": final,
    }
