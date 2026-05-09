"""FinFlowRL Neural Network Models.

This sub-package provides the core neural network components for the
FinFlowRL imitation-reinforcement learning framework:

1. **MeanFlowPolicy** – Core flow-matching policy (Geng et al. 2025) that
   learns a conditional average-velocity field for one-step action generation.
2. **FiLMLayer** – Feature-wise Linear Modulation layer (Perez et al. 2018)
   for conditioning the flow on market state.
3. **NoisePolicy** – Lightweight Gaussian noise policy for FlowRL stage-2
   fine-tuning via PPO, dramatically reducing trainable parameters (~84%).

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from .meanflow import MeanFlowPolicy
from .film import FiLMLayer
from .noise_policy import NoisePolicy

__all__ = ["MeanFlowPolicy", "FiLMLayer", "NoisePolicy"]
