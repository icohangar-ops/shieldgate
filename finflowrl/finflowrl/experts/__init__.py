"""FinFlowRL expert strategies for market-making.

This sub-package provides four market-making expert strategies used as
demonstration policies in the FinFlowRL imitation-reinforcement learning
framework:

1. **Avellaneda-Stoikov (AS)** – Classical closed-form optimal quoting
   with symmetric half-spreads under exponential risk aversion.
2. **GLFT** – Guéant-Lehalle-Fernandez-Tapia extension with asymmetric
   arrival intensities and closed-form approximation.
3. **GLFT-Drift** – GLFT augmented with an auto-estimated price-drift
   term for trending markets.
4. **PPO-Expert** – A lightweight NumPy MLP policy network that can be
   loaded with weights from a PPO training checkpoint.

All experts share a common interface through the :class:`Expert` abstract
base class and the :class:`MarketState` / :class:`ExpertAction` data
containers.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from .base import Expert, ExpertAction, MarketState, clip_spread
from .avellaneda_stoikov import AvellanedaStoikovExpert
from .glft import GLFTExpert
from .glft_drift import GLFTDriftExpert
from .ppo_expert import PPOExpert

__all__ = [
    "Expert",
    "ExpertAction",
    "MarketState",
    "clip_spread",
    "AvellanedaStoikovExpert",
    "GLFTExpert",
    "GLFTDriftExpert",
    "PPOExpert",
]
