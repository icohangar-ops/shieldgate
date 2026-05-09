"""Tests for evaluation metrics."""

import numpy as np
import pytest

from finflowrl.evaluation.metrics import (
    EvaluationResult,
    compare_strategies,
    compute_max_drawdown,
    compute_pnl,
    compute_sharpe_ratio,
    evaluate_strategy,
)


class TestComputePnl:
    def test_normal_returns(self):
        returns = np.array([1.0, -0.5, 2.0, -1.0])
        assert compute_pnl(returns) == pytest.approx(1.5)

    def test_all_positive(self):
        returns = np.array([1.0, 2.0, 3.0])
        assert compute_pnl(returns) == pytest.approx(6.0)

    def test_empty(self):
        assert compute_pnl(np.array([])) == 0.0

    def test_none(self):
        assert compute_pnl(None) == 0.0

    def test_single_element(self):
        assert compute_pnl(np.array([5.5])) == pytest.approx(5.5)


class TestComputeSharpeRatio:
    def test_positive_mean(self):
        returns = np.random.normal(0.01, 0.02, 1000)
        sr = compute_sharpe_ratio(returns)
        assert sr > 0

    def test_negative_mean(self):
        returns = np.random.normal(-0.01, 0.02, 1000)
        sr = compute_sharpe_ratio(returns)
        assert sr < 0

    def test_zero_std(self):
        returns = np.ones(100)
        assert compute_sharpe_ratio(returns) == 0.0

    def test_single_element(self):
        assert compute_sharpe_ratio(np.array([1.0])) == 0.0

    def test_two_elements(self):
        # Two elements with some variance
        sr = compute_sharpe_ratio(np.array([1.0, 2.0]))
        assert sr != 0.0

    def test_with_risk_free_rate(self):
        returns = np.random.normal(0.01, 0.02, 1000)
        sr0 = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        sr1 = compute_sharpe_ratio(returns, risk_free_rate=0.05)
        assert sr1 < sr0  # Higher risk-free reduces SR


class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        returns = np.ones(100)
        assert compute_max_drawdown(returns) == 0.0

    def test_drawdown_exists(self):
        returns = np.array([5.0, -2.0, -1.0, 1.0, 3.0])
        mdd = compute_max_drawdown(returns)
        assert mdd > 0

    def test_empty(self):
        assert compute_max_drawdown(np.array([])) == 0.0

    def test_monotonic_increase(self):
        returns = np.arange(1.0, 101.0)
        assert compute_max_drawdown(returns) == 0.0

    def test_monotonic_decrease(self):
        returns = -np.arange(1.0, 101.0)
        mdd = compute_max_drawdown(returns)
        # All negative returns means peak = 0 at start, so no drawdown from peak
        # Actually cumulative starts negative immediately
        assert mdd >= 0


class TestEvaluateStrategy:
    def test_basic(self):
        returns = np.random.normal(0.001, 0.02, 500)
        result = evaluate_strategy(returns, "Test")
        assert isinstance(result, EvaluationResult)
        assert result.strategy_name == "Test"
        assert isinstance(result.total_pnl, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown >= 0
        assert 0 <= result.win_rate <= 1

    def test_with_market_condition(self):
        returns = np.random.normal(0.001, 0.02, 500)
        result = evaluate_strategy(returns, "Test", market_condition="HighVol")
        assert result.market_condition == "HighVol"

    def test_summary(self):
        returns = np.random.normal(0.001, 0.02, 500)
        result = evaluate_strategy(returns, "MyStrategy")
        s = result.summary()
        assert "MyStrategy" in s
        assert "PnL" in s
        assert "SR" in s
        assert "MDD" in s


class TestCompareStrategies:
    def test_empty(self):
        assert "No results" in compare_strategies([])

    def test_single(self):
        returns = np.random.normal(0.001, 0.02, 500)
        result = evaluate_strategy(returns, "A")
        table = compare_strategies([result])
        assert "A" in table

    def test_multiple(self):
        np.random.seed(42)
        results = [
            evaluate_strategy(np.random.normal(0.002, 0.01, 500), "Alpha"),
            evaluate_strategy(np.random.normal(0.001, 0.02, 500), "Beta"),
            evaluate_strategy(np.random.normal(-0.001, 0.03, 500), "Gamma"),
        ]
        table = compare_strategies(results)
        assert "Alpha" in table
        assert "Beta" in table
        assert "Gamma" in table
        assert "Best PnL" in table
        assert "Best SR" in table
        assert "Best MDD" in table
