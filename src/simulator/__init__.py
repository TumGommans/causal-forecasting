"""Simulation module for generating retail discount DGPs.

This module provides functionality to generate four different Data Generating Processes (DGPs)
for evaluating causal inference methods in retail settings with discounts, sparsity, and interference.
"""

from .generator import SimulationGenerator
from .core.data_structures import SimulationData

__all__ = ['SimulationGenerator', 'SimulationData']
__version__ = '1.0.0'