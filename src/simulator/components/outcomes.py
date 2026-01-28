"""Outcome generation utilities (unused in current implementation but useful for extensions)."""

import numpy as np

from ..core.utils import log_normal_sample, zero_inflated_log_normal_sample


class OutcomeGenerator:
    """Utilities for generating outcomes (for potential future use)."""
    
    @staticmethod
    def log_normal(mu: np.ndarray, sigma: float) -> np.ndarray:
        """Generate log-normal outcomes."""
        return log_normal_sample(mu, sigma)
    
    @staticmethod
    def zero_inflated_log_normal(
        pi: np.ndarray, 
        mu: np.ndarray, 
        sigma: float
    ) -> np.ndarray:
        """Generate zero-inflated log-normal outcomes."""
        return zero_inflated_log_normal_sample(pi, mu, sigma)