"""Tests for expert strategies."""

import numpy as np
import pytest

from finflowrl.experts import (
    Expert,
    AvellanedaStoikovExpert,
    GLFTExpert,
    GLFTDriftExpert,
    PPOExpert,
)
from finflowrl.experts.base import ExpertAction, MarketState, clip_spread


def make_market_state(**overrides):
    """Create a default MarketState with optional overrides."""
    defaults = dict(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.2,
        sell_intensity=0.8,
        price_change=0.01,
        volatility=0.02,
        time_remaining=0.75,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


class TestMarketState:
    def test_creation(self):
        ms = make_market_state()
        assert ms.mid_price == 100.0
        assert ms.inventory == 0
        assert ms.time_remaining == 0.75

    def test_default_time_remaining(self):
        ms = MarketState(
            mid_price=100, inventory=0, spread=0.05,
            buy_intensity=1.0, sell_intensity=1.0,
            price_change=0.0, volatility=0.02,
        )
        assert ms.time_remaining == 1.0


class TestClipSpread:
    def test_within_range(self):
        assert clip_spread(0.5) == 0.5

    def test_below_minimum(self):
        assert clip_spread(-0.5) == 0.01

    def test_above_maximum(self):
        assert clip_spread(5.0) == 2.0

    def test_custom_range(self):
        assert clip_spread(0.5, lo=0.1, hi=1.0) == 0.5
        assert clip_spread(0.05, lo=0.1, hi=1.0) == 0.1
        assert clip_spread(1.5, lo=0.1, hi=1.0) == 1.0


class TestAvellanedaStoikovExpert:
    def test_creation(self):
        expert = AvellanedaStoikovExpert()
        assert expert.name() == "Avellaneda-Stoikov"

    def test_act_neutral_inventory(self):
        expert = AvellanedaStoikovExpert(gamma=0.1, k=1.5)
        state = make_market_state(inventory=0)
        action = expert.act(state)
        assert isinstance(action, ExpertAction)
        assert 0.01 <= action.delta_bid <= 2.0
        assert 0.01 <= action.delta_ask <= 2.0

    def test_act_long_inventory(self):
        expert = AvellanedaStoikovExpert(gamma=0.1, k=1.5)
        state_long = make_market_state(inventory=20, time_remaining=0.5)
        action = expert.act(state_long)
        # Long inventory should widen bid (higher delta_bid) and tighten ask
        state_neutral = make_market_state(inventory=0, time_remaining=0.5)
        action_neutral = expert.act(state_neutral)
        assert action.delta_bid >= action_neutral.delta_bid

    def test_act_short_inventory(self):
        expert = AvellanedaStoikovExpert(gamma=0.1, k=1.5)
        state_short = make_market_state(inventory=-15, time_remaining=0.5)
        state_neutral = make_market_state(inventory=0, time_remaining=0.5)
        action_short = expert.act(state_short)
        action_neutral = expert.act(state_neutral)
        # Short inventory: tighten bid, widen ask
        assert action_short.delta_ask >= action_neutral.delta_ask

    def test_act_zero_volatility(self):
        expert = AvellanedaStoikovExpert()
        state = make_market_state(volatility=0.0)
        action = expert.act(state)
        assert 0.01 <= action.delta_bid <= 2.0
        assert 0.01 <= action.delta_ask <= 2.0

    def test_is_expert(self):
        expert = AvellanedaStoikovExpert()
        assert isinstance(expert, Expert)

    def test_repr(self):
        expert = AvellanedaStoikovExpert(gamma=0.2, k=3.0)
        r = repr(expert)
        assert "AvellanedaStoikovExpert" in r
        assert "gamma=0.2" in r

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            AvellanedaStoikovExpert(gamma=-0.1)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k"):
            AvellanedaStoikovExpert(k=0.0)


class TestGLFTExpert:
    def test_creation(self):
        expert = GLFTExpert()
        assert expert.name() == "GLFT"

    def test_act(self):
        expert = GLFTExpert(gamma=0.1, kappa=1.5)
        state = make_market_state()
        action = expert.act(state)
        assert 0.01 <= action.delta_bid <= 2.0
        assert 0.01 <= action.delta_ask <= 2.0

    def test_is_expert(self):
        expert = GLFTExpert()
        assert isinstance(expert, Expert)

    def test_asymmetric_intensities(self):
        expert = GLFTExpert()
        # Buy-heavy market
        state_buy_heavy = make_market_state(buy_intensity=5.0, sell_intensity=0.5)
        state_sell_heavy = make_market_state(buy_intensity=0.5, sell_intensity=5.0)
        a_buy = expert.act(state_buy_heavy)
        a_sell = expert.act(state_sell_heavy)
        # Results should differ due to asymmetric intensities
        # (not necessarily in a specific direction, but different)
        assert a_buy.delta_bid != a_sell.delta_bid or a_buy.delta_ask != a_sell.delta_ask


class TestGLFTDriftExpert:
    def test_creation(self):
        expert = GLFTDriftExpert()
        assert "Drift" in expert.name() or "GLFT" in expert.name()

    def test_act(self):
        expert = GLFTDriftExpert()
        state = make_market_state()
        action = expert.act(state)
        assert 0.01 <= action.delta_bid <= 2.0
        assert 0.01 <= action.delta_ask <= 2.0

    def test_is_expert(self):
        expert = GLFTDriftExpert()
        assert isinstance(expert, Expert)


class TestPPOExpert:
    def test_creation(self):
        expert = PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=42)
        assert isinstance(expert, Expert)

    def test_act(self):
        expert = PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=42)
        state = make_market_state()
        action = expert.act(state)
        assert isinstance(action, ExpertAction)
        assert 0.01 <= action.delta_bid <= 2.0
        assert 0.01 <= action.delta_ask <= 2.0

    def test_save_load_weights(self, tmp_path):
        expert = PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=42)
        state = make_market_state()

        weight_file = str(tmp_path / "ppo_weights.npz")
        expert.save_weights(weight_file)

        loaded = PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=0)
        loaded.load_weights(weight_file)

        a1 = expert.act(state)
        a2 = loaded.act(state)
        assert a1.delta_bid == pytest.approx(a2.delta_bid)
        assert a1.delta_ask == pytest.approx(a2.delta_ask)

    def test_reset(self):
        expert = PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=42)
        expert.reset()  # should not raise


class TestAllExperts:
    def test_all_experts_are_expert_subclass(self):
        experts = [
            AvellanedaStoikovExpert(),
            GLFTExpert(),
            GLFTDriftExpert(),
            PPOExpert(obs_dim=14, action_dim=2, hidden_dim=32, seed=42),
        ]
        for e in experts:
            assert isinstance(e, Expert)

    def test_all_experts_produce_valid_actions(self):
        state = make_market_state()
        experts = [
            AvellanedaStoikovExpert(),
            GLFTExpert(),
            GLFTDriftExpert(),
        ]
        for e in experts:
            action = e.act(state)
            assert 0.01 <= action.delta_bid <= 2.0, f"{e.name()} bid out of range"
            assert 0.01 <= action.delta_ask <= 2.0, f"{e.name()} ask out of range"
