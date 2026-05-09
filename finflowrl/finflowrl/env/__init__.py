"""FinFlowRL Environment Module.

Provides a realistic market microstructure simulator and a Gym-style
high-frequency trading (HFT) market-making environment for reinforcement-
learning research, following the formulation in:

    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964).
"""

from .market_simulator import MarketSimulator
from .hft_env import HFTEnv

__all__ = ["MarketSimulator", "HFTEnv"]
