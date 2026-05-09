"""FinFlowRL Evaluation Module.

Provides financial evaluation metrics for comparing market-making strategies,
including PnL, Sharpe ratio, maximum drawdown, and comprehensive evaluation
result containers.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from .metrics import (
    EvaluationResult,
    compute_max_drawdown,
    compute_pnl,
    compute_sharpe_ratio,
    compare_strategies,
    evaluate_strategy,
)

__all__ = [
    "EvaluationResult",
    "compute_pnl",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "evaluate_strategy",
    "compare_strategies",
]
