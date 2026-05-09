"""Financial evaluation metrics for comparing market-making strategies.

Provides composable metrics (PnL, Sharpe ratio, max drawdown, win rate) and a
convenience wrapper :func:`evaluate_strategy` that returns an
:class:`EvaluationResult` dataclass with a formatted summary string.
Multiple results can be compared side-by-side via :func:`compare_strategies`.

All metrics operate on NumPy arrays of per-step returns (or rewards).

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ======================================================================
# Data containers
# ======================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results of a single strategy.

    Attributes
    ----------
    strategy_name : str
        Human-readable identifier for the strategy.
    total_pnl : float
        Cumulative profit-and-loss over the evaluation horizon.
    sharpe_ratio : float
        Annualised Sharpe ratio (default 252 trading periods/year).
    max_drawdown : float
        Maximum peak-to-trough drawdown as a percentage (>= 0).
    win_rate : float
        Fraction of positive-return steps in [0, 1].
    num_trades : int
        Number of steps with non-zero returns (proxy for trade count).
    avg_pnl_per_trade : float
        Average PnL per trade (total_pnl / max(num_trades, 1)).
    pnl_std : float
        Standard deviation of per-step returns.
    market_condition : str
        Optional free-text label describing the market regime.
    """

    strategy_name: str
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_pnl_per_trade: float
    pnl_std: float
    market_condition: str = ""

    def summary(self) -> str:
        """Return a single-line formatted summary string.

        Returns
        -------
        str
            Formatted string, e.g. ``"Strategy             | PnL:   123.45 | SR:  1.234 | MDD:   5.67% | Trades: 500"``.
        """
        return (
            f"{self.strategy_name:20s} | "
            f"PnL: {self.total_pnl:8.2f} | "
            f"SR: {self.sharpe_ratio:6.3f} | "
            f"MDD: {self.max_drawdown:7.2f}% | "
            f"Trades: {self.num_trades}"
        )

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(\n"
            f"  strategy={self.strategy_name!r},\n"
            f"  total_pnl={self.total_pnl:.4f},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.4f},\n"
            f"  max_drawdown={self.max_drawdown:.4f}%,\n"
            f"  win_rate={self.win_rate:.4f},\n"
            f"  num_trades={self.num_trades},\n"
            f"  avg_pnl_per_trade={self.avg_pnl_per_trade:.6f},\n"
            f"  pnl_std={self.pnl_std:.6f},\n"
            f"  market_condition={self.market_condition!r}\n"
            f")"
        )


# ======================================================================
# Core metrics
# ======================================================================

def compute_pnl(returns: np.ndarray) -> float:
    """Compute total PnL from a return series.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-step returns (or rewards).

    Returns
    -------
    float
        Sum of all returns.  Returns ``0.0`` for empty input.
    """
    if returns is None or len(returns) == 0:
        return 0.0
    return float(np.sum(np.asarray(returns, dtype=np.float64)))


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio.

    .. math::

        SR = \\frac{\\bar{r} - r_f / P}{\\sigma_r} \\cdot \\sqrt{P}

    where :math:`\\bar{r}` is the mean per-step return,
    :math:`r_f` is the annualised risk-free rate,
    :math:`P` is the number of periods per year, and
    :math:`\\sigma_r` is the standard deviation of per-step returns.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-step returns.
    risk_free_rate : float
        Annualised risk-free rate.  Default ``0.0``.
    periods_per_year : int
        Number of trading periods per year for annualisation.
        Default ``252`` (daily bars).

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns ``0.0`` if the input has
        fewer than 2 elements or zero standard deviation.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-10:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown as a percentage.

    Drawdown is defined as the peak-to-trough decline in the cumulative
    return curve, expressed as a percentage of the peak value.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-step returns.

    Returns
    -------
    float
        Maximum drawdown in percent (>= 0).  Returns ``0.0`` for
        empty input or when the cumulative return never exceeds zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_peak = float(np.max(peak))
    if max_peak < 1e-10:
        return 0.0
    return float(np.max(drawdown) / max_peak * 100.0)


# ======================================================================
# Composite evaluation
# ======================================================================

def evaluate_strategy(
    returns: np.ndarray,
    strategy_name: str = "Strategy",
    risk_free_rate: float = 0.0,
    market_condition: str = "",
) -> EvaluationResult:
    """Full evaluation of a strategy from a return series.

    Computes all standard metrics and returns them in an
    :class:`EvaluationResult` container.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-step returns (or rewards).
    strategy_name : str
        Human-readable strategy identifier.  Default ``"Strategy"``.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe computation.
        Default ``0.0``.
    market_condition : str
        Optional label describing the market regime.

    Returns
    -------
    EvaluationResult
        Container with all computed metrics.
    """
    returns = np.asarray(returns, dtype=np.float64)

    pnl = compute_pnl(returns)
    sr = compute_sharpe_ratio(returns, risk_free_rate)
    mdd = compute_max_drawdown(returns)

    # Count "trades" as steps with non-negligible absolute return
    trades = int(np.sum(np.abs(returns) > 1e-10))

    # Win rate: fraction of positive-return steps
    win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0

    return EvaluationResult(
        strategy_name=strategy_name,
        total_pnl=pnl,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        win_rate=win_rate,
        num_trades=trades,
        avg_pnl_per_trade=pnl / max(trades, 1),
        pnl_std=float(np.std(returns)),
        market_condition=market_condition,
    )


def compare_strategies(results: List[EvaluationResult]) -> str:
    """Format a comparison table of multiple strategy evaluation results.

    Sorts by total PnL descending and renders a fixed-width table with
    all key metrics.

    Parameters
    ----------
    results : list of EvaluationResult
        Evaluation results for each strategy to compare.

    Returns
    -------
    str
        A multi-line formatted comparison table string.
    """
    if not results:
        return "No results to compare."

    # Sort by total PnL descending
    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    # Header
    header = (
        f"{'Strategy':<20s} | {'PnL':>9s} | {'SR':>7s} | "
        f"{'MDD':>8s} | {'WinRate':>7s} | {'Trades':>7s} | "
        f"{'AvgPnL':>9s}"
    )
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "  Strategy Comparison (sorted by PnL)",
        "=" * len(header),
        header,
        sep,
    ]
    for r in sorted_results:
        line = (
            f"{r.strategy_name:<20s} | {r.total_pnl:9.2f} | "
            f"{r.sharpe_ratio:7.3f} | {r.max_drawdown:7.2f}% | "
            f"{r.win_rate:6.1%}  | {r.num_trades:7d} | "
            f"{r.avg_pnl_per_trade:9.4f}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append("")

    # Best-per-strategy summary
    if len(sorted_results) > 1:
        best = sorted_results[0]
        lines.append(f"  Best PnL:  {best.strategy_name} ({best.total_pnl:.2f})")
        best_sr = max(sorted_results, key=lambda r: r.sharpe_ratio)
        lines.append(f"  Best SR:   {best_sr.strategy_name} ({best_sr.sharpe_ratio:.3f})")
        best_mdd = min(sorted_results, key=lambda r: r.max_drawdown)
        lines.append(f"  Best MDD:  {best_mdd.strategy_name} ({best_mdd.max_drawdown:.2f}%)")
        lines.append("=" * len(header))

    return "\n".join(lines)


# ======================================================================
# Main – smoke test with random returns
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("  FinFlowRL Evaluation Metrics – Smoke Test")
    print("=" * 70)

    # --- Test 1: Individual metrics ---
    print("\n--- Test 1: Individual metrics ---")
    returns = np.random.normal(0.001, 0.02, size=1000)

    pnl = compute_pnl(returns)
    sr = compute_sharpe_ratio(returns)
    mdd = compute_max_drawdown(returns)

    print(f"  compute_pnl:           {pnl:.4f}")
    print(f"  compute_sharpe_ratio:  {sr:.4f}")
    print(f"  compute_max_drawdown:  {mdd:.2f}%")
    assert isinstance(pnl, float), "PnL should be a float"
    assert isinstance(sr, float), "Sharpe ratio should be a float"
    assert isinstance(mdd, float), "MDD should be a float"
    assert mdd >= 0.0, "MDD should be non-negative"
    print("  [PASS]")

    # --- Test 2: Edge cases ---
    print("\n--- Test 2: Edge cases ---")
    assert compute_pnl(np.array([])) == 0.0, "Empty array PnL should be 0"
    assert compute_sharpe_ratio(np.array([1.0])) == 0.0, "Single element SR should be 0"
    assert compute_max_drawdown(np.array([])) == 0.0, "Empty array MDD should be 0"
    # Constant returns → zero variance → SR = 0
    assert compute_sharpe_ratio(np.ones(100)) == 0.0, "Constant returns SR should be 0"
    # All positive returns → MDD = 0
    assert compute_max_drawdown(np.ones(50)) == 0.0, "All-positive returns MDD should be 0"
    print("  [PASS]")

    # --- Test 3: evaluate_strategy ---
    print("\n--- Test 3: evaluate_strategy ---")
    result = evaluate_strategy(returns, strategy_name="Random", market_condition="Normal")
    print(f"  {result.summary()}")
    assert result.total_pnl == pnl, "PnL mismatch"
    assert result.sharpe_ratio == sr, "SR mismatch"
    assert result.max_drawdown == mdd, "MDD mismatch"
    assert 0.0 <= result.win_rate <= 1.0, "Win rate out of [0, 1]"
    assert result.num_trades > 0, "Should have trades with random returns"
    print(f"  {result}")
    print("  [PASS]")

    # --- Test 4: compare_strategies ---
    print("\n--- Test 4: compare_strategies ---")
    r1 = evaluate_strategy(np.random.normal(0.002, 0.01, 500), "Strategy-A")
    r2 = evaluate_strategy(np.random.normal(0.001, 0.03, 500), "Strategy-B")
    r3 = evaluate_strategy(np.random.normal(-0.001, 0.02, 500), "Strategy-C")

    table = compare_strategies([r1, r2, r3])
    print(table)
    assert "Strategy-A" in table, "Strategy-A should be in table"
    assert "Best PnL" in table, "Best PnL line should be in table"
    assert "Best SR" in table, "Best SR line should be in table"
    print("  [PASS]")

    # --- Test 5: Reproducibility ---
    print("\n--- Test 5: Reproducibility ---")
    np.random.seed(123)
    a = np.random.normal(0.0, 0.01, 200)
    np.random.seed(123)
    b = np.random.normal(0.0, 0.01, 200)
    assert np.allclose(a, b), "Seeded random should be reproducible"
    assert compute_pnl(a) == compute_pnl(b), "Same data → same PnL"
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("  metrics.py OK – all tests passed.")
    print("=" * 70)
