"""Tests for market simulator and HFT environment."""

import numpy as np
import pytest

from finflowrl.env import MarketSimulator, HFTEnv


class TestMarketSimulator:
    """Tests for MarketSimulator."""

    def test_creation_default(self):
        sim = MarketSimulator()
        assert sim.S0 == 100.0
        assert sim.sigma == 0.1
        assert sim.H == 0.5

    def test_creation_custom(self):
        sim = MarketSimulator(S0=50.0, sigma=0.3, H=0.7, seed=99)
        assert sim.S0 == 50.0
        assert sim.sigma == 0.3
        assert sim.H == 0.7

    def test_reset(self):
        sim = MarketSimulator(seed=42)
        state = sim.reset()
        assert isinstance(state, dict)
        expected_keys = {"mid_price", "bid_price", "ask_price", "spread",
                         "buy_intensity", "sell_intensity", "buy_orders",
                         "sell_orders", "jump_occurred", "inventory_delta"}
        assert set(state.keys()) == expected_keys
        assert state["mid_price"] == sim.S0

    def test_step_returns_valid_state(self):
        sim = MarketSimulator(seed=42)
        sim.reset()
        state = sim.step()
        assert isinstance(state, dict)
        assert state["mid_price"] > 0
        assert state["bid_price"] < state["ask_price"]
        assert state["spread"] > 0
        assert state["buy_intensity"] >= 0
        assert state["sell_intensity"] >= 0

    def test_step_price_evolution(self):
        sim = MarketSimulator(seed=42)
        sim.reset()
        prices = []
        for _ in range(100):
            state = sim.step()
            prices.append(state["mid_price"])
        # Price should change over time (not constant)
        assert max(prices) != min(prices)
        # Price should stay positive
        assert all(p > 0 for p in prices)

    def test_trajectory(self):
        sim = MarketSimulator(seed=42)
        traj = sim.generate_trajectory(50)
        assert len(traj) == 50
        assert all(isinstance(s, dict) for s in traj)

    def test_reproducibility(self):
        sim1 = MarketSimulator(seed=123)
        sim2 = MarketSimulator(seed=123)
        sim1.reset()
        sim2.reset()
        for _ in range(10):
            s1 = sim1.step()
            s2 = sim2.step()
            assert s1["mid_price"] == pytest.approx(s2["mid_price"])

    def test_fbm_covariance_symmetric(self):
        sim = MarketSimulator(H=0.5)
        cov = sim._build_fbm_covariance(10)
        assert cov.shape == (10, 10)
        assert np.allclose(cov, cov.T)
        # For H=0.5, should be close to identity (standard BM)
        diag = np.diag(cov)
        off_diag = cov - np.diag(diag)
        assert np.max(np.abs(off_diag)) < 1e-6

    def test_hawkes_intensity_builds(self):
        sim = MarketSimulator(mu_a=10.0, alpha_aa=5.0, beta=10.0, seed=42)
        sim.reset()
        intensities = []
        for _ in range(50):
            sim.step()
            intensities.append(sim._buy_intensity)
        # With self-excitation, intensity should sometimes exceed baseline
        assert max(intensities) >= sim.mu_a


class TestHFTEnv:
    """Tests for HFTEnv."""

    def _make_env(self, max_steps=100, seed=42):
        sim = MarketSimulator(seed=seed)
        return HFTEnv(sim, max_steps=max_steps)

    def test_creation(self):
        env = self._make_env()
        assert env.max_steps == 100
        assert env.T_obs == 2
        assert env.obs_dim == 14  # T_obs * 7

    def test_action_space(self):
        env = self._make_env()
        space = env.action_space
        assert space["shape"] == (2,)
        assert space["low"] == [0.01, 0.01]
        assert space["high"] == [2.0, 2.0]

    def test_observation_space(self):
        env = self._make_env()
        space = env.observation_space
        assert space["shape"] == (14,)

    def test_reset(self):
        env = self._make_env()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (14,)
        assert np.isfinite(obs).all()

    def test_step(self):
        env = self._make_env()
        env.reset()
        obs, reward, done, info = env.step(np.array([0.1, 0.1]))
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (14,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self):
        env = self._make_env()
        env.reset()
        # Extreme actions should be clipped
        _, _, _, info = env.step(np.array([-10.0, 100.0]))
        assert info["bid_price"] > 0
        assert info["ask_price"] > info["bid_price"]

    def test_episode_termination(self):
        env = self._make_env(max_steps=20)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, info = env.step(np.array([0.1, 0.1]))
            steps += 1
        # steps can be max_steps+1 due to reset() taking an initial step
        assert steps <= 21

    def test_info_keys(self):
        env = self._make_env()
        env.reset()
        _, _, _, info = env.step(np.array([0.1, 0.1]))
        expected_keys = {"step", "inventory", "cash", "mid_price",
                         "filled_buy", "filled_sell", "bid_price",
                         "ask_price", "inventory_breach"}
        assert expected_keys.issubset(info.keys())

    def test_multiple_episodes(self):
        env = self._make_env(max_steps=10)
        for _ in range(5):
            obs = env.reset()
            assert obs.shape == (14,)
            done = False
            while not done:
                _, _, done, _ = env.step(np.array([0.1, 0.1]))
