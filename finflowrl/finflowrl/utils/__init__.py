"""FinFlowRL Utilities Module.

Provides configuration management, expert data generation, and dataset
persistence utilities for the FinFlowRL training pipeline.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from .config import Config, load_config
from .data import generate_expert_demonstrations, load_dataset, save_dataset

__all__ = [
    "Config",
    "load_config",
    "generate_expert_demonstrations",
    "save_dataset",
    "load_dataset",
]
