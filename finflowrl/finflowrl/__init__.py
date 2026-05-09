"""FinFlowRL: Imitation-Reinforcement Learning Framework for Adaptive Stochastic Control in Finance.

Based on: Li et al. (2025) "FinFlowRL" arXiv:2509.17964

Two-stage training pipeline:
    - Stage 1: MeanFlow pre-training on expert demonstrations (flow matching)
    - Stage 2: FlowRL fine-tuning via PPO in noise space (~84% param reduction)

Sub-modules
-----------
    - ``env``: Market simulator and HFT gym-style environment
    - ``experts``: Avellaneda-Stoikov, GLFT, GLFT-Drift, PPO expert strategies
    - ``models``: MeanFlowPolicy, FiLMLayer, NoisePolicy neural networks
    - ``evaluation``: Financial metrics (PnL, Sharpe ratio, max drawdown)
    - ``training``: Pre-training and fine-tuning pipelines
    - ``utils``: Configuration management and data generation utilities
"""

__version__ = "0.1.0"
__author__ = "FinFlowRL Team"

from .env import HFTEnv, MarketSimulator
from .evaluation import (
    EvaluationResult,
    compute_max_drawdown,
    compute_pnl,
    compute_sharpe_ratio,
    evaluate_strategy,
)
from .experts import (
    AvellanedaStoikovExpert,
    Expert,
    GLFTDriftExpert,
    GLFTExpert,
    PPOExpert,
)
from .models import FiLMLayer, MeanFlowPolicy, NoisePolicy

__all__ = [
    # Version
    "__version__",
    # Environment
    "MarketSimulator",
    "HFTEnv",
    # Experts
    "Expert",
    "AvellanedaStoikovExpert",
    "GLFTExpert",
    "GLFTDriftExpert",
    "PPOExpert",
    # Models
    "MeanFlowPolicy",
    "FiLMLayer",
    "NoisePolicy",
    # Evaluation
    "compute_pnl",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "evaluate_strategy",
    "EvaluationResult",
]
