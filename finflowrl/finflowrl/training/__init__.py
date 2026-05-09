"""FinFlowRL Training Module.

Two-stage training pipeline following the FinFlowRL framework:

1. **Stage 1 – MeanFlow Pre-training** (:class:`MeanFlowPretrainer`):
   Learns a flow-matching model that maps Gaussian noise to expert actions,
   conditioned on market state, via the MeanFlow consistency loss.

2. **Stage 2 – FlowRL Fine-tuning** (:class:`FlowRLFinetuner`):
   Freezes the pre-trained MeanFlow model and trains a lightweight noise
   policy via PPO in noise space, achieving ~84% parameter reduction.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from .pretrain import MeanFlowPretrainer
from .finetune import FlowRLFinetuner

__all__ = ["MeanFlowPretrainer", "FlowRLFinetuner"]
