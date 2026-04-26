"""Core data structure for tabular MDPs."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TabularMDP:
    """Tabular MDP extracted from a Gymnasium environment.

    Attributes:
        nS: Number of states.
        nA: Number of actions.
        P: Transition kernel, shape [nS, nA, nS].
        terminal: Boolean mask for terminal states, shape [nS].
        start: Initial state distribution, shape [nS].
        legal: Boolean mask for legal actions, shape [nS, nA].
    """
    nS: int
    nA: int
    P: np.ndarray
    terminal: np.ndarray
    start: np.ndarray
    legal: np.ndarray

    def nonterminal_indices(self) -> np.ndarray:
        return np.where(~self.terminal)[0]

    def terminal_indices(self) -> np.ndarray:
        return np.where(self.terminal)[0]

    def validate(self) -> None:
        """Run basic consistency checks."""
        assert self.P.shape == (self.nS, self.nA, self.nS)
        assert self.terminal.shape == (self.nS,)
        assert self.start.shape == (self.nS,)
        assert self.legal.shape == (self.nS, self.nA)
        row_sums = self.P.sum(axis=2)
        active = self.legal & ~self.terminal[:, None]
        max_err = np.abs(row_sums[active] - 1.0).max() if active.any() else 0.0
        assert max_err < 1e-10, f"Transition rows don't sum to 1: max error {max_err}"
