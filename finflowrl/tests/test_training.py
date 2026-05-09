"""Tests for training components and utilities."""

import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from finflowrl.training import MeanFlowPretrainer, FlowRLFinetuner
from finflowrl.utils import Config, load_config
from finflowrl.utils.config import (
    FinetuneConfig,
    MarketConfig,
    ModelConfig,
    PretrainConfig,
)
from finflowrl.training.pretrain import ExpertDataset


class TestExpertDataset:
    def test_creation(self):
        states = np.random.randn(100, 2, 14).astype(np.float32)
        actions = np.random.rand(100, 8, 2).astype(np.float32) * 1.99 + 0.01
        ds = ExpertDataset(states, actions)
        assert len(ds) == 100

    def test_getitem(self):
        states = np.random.randn(10, 2, 14).astype(np.float32)
        actions = np.random.rand(10, 8, 2).astype(np.float32) * 1.99 + 0.01
        ds = ExpertDataset(states, actions)
        s, a = ds[0]
        assert s.shape == (2, 14)
        assert a.shape == (8, 2)

    def test_dataloader(self):
        states = np.random.randn(32, 2, 14).astype(np.float32)
        actions = np.random.rand(32, 8, 2).astype(np.float32) * 1.99 + 0.01
        ds = ExpertDataset(states, actions)
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        batch_s, batch_a = next(iter(dl))
        assert batch_s.shape[0] <= 8
        assert batch_a.shape[0] <= 8


class TestConfig:
    def test_default(self):
        config = Config()
        assert config.market.sigma == 0.1
        assert config.model.state_dim == 14
        assert config.seed == 42
        assert config.pretrain.learning_rate == 3e-4

    def test_yaml_roundtrip(self):
        config = Config()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        try:
            config.to_yaml(path)
            loaded = Config.from_yaml(path)
            assert loaded.market.sigma == config.market.sigma
            assert loaded.model.T_pred == config.model.T_pred
            assert loaded.seed == config.seed
            assert loaded.finetune.gamma == config.finetune.gamma
        finally:
            import os
            os.unlink(path)

    def test_from_dict_overrides(self):
        config = Config.from_dict({
            "market": {"sigma": 0.3, "H": 0.7},
            "model": {"hidden_dim": 256},
            "seed": 99,
        })
        assert config.market.sigma == 0.3
        assert config.market.H == 0.7
        assert config.model.hidden_dim == 256
        assert config.seed == 99
        assert config.market.S0 == 100.0  # default preserved

    def test_to_dict(self):
        config = Config()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "market" in d
        assert "model" in d
        assert d["market"]["sigma"] == 0.1

    def test_summary(self):
        config = Config()
        s = config.summary()
        assert "FinFlowRL" in s
        assert "sigma=0.1" in s
        assert "T_pred=8" in s

    def test_load_config_helper(self):
        config = Config()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = f.name
        try:
            config.to_yaml(path)
            loaded = load_config(path)
            assert loaded.seed == config.seed
        finally:
            import os
            os.unlink(path)

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path.yaml")


class TestMeanFlowPretrainer:
    def test_creation(self):
        from finflowrl.models import MeanFlowPolicy
        model = MeanFlowPolicy(
            state_dim=14, action_dim=2, noise_dim=8,
            T_obs=2, T_pred=4, T_exec=2,
            hidden_dim=32, num_layers=2,
        )
        pretrainer = MeanFlowPretrainer(
            model=model,
            learning_rate=1e-3,
            batch_size=16,
        )
        assert pretrainer is not None

    def test_train_step(self):
        from finflowrl.models import MeanFlowPolicy
        model = MeanFlowPolicy(
            state_dim=14, action_dim=2, noise_dim=8,
            T_obs=2, T_pred=4, T_exec=2,
            hidden_dim=32, num_layers=2,
        )
        pretrainer = MeanFlowPretrainer(
            model=model,
            learning_rate=1e-3,
            batch_size=16,
        )
        states = torch.randn(8, 2, 14)
        actions = torch.rand(8, 4, 2) * 1.99 + 0.01
        metrics = pretrainer.train_step(states, actions)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert np.isfinite(metrics["loss"])


class TestFlowRLFinetuner:
    def test_creation(self):
        from finflowrl.models import MeanFlowPolicy, NoisePolicy
        from finflowrl.env import MarketSimulator, HFTEnv
        meanflow = MeanFlowPolicy(
            state_dim=14, action_dim=2, noise_dim=8,
            T_obs=2, T_pred=4, T_exec=2,
            hidden_dim=32, num_layers=2,
        )
        noise_policy = NoisePolicy(state_dim=28, noise_dim=32, hidden_dim=32)
        sim = MarketSimulator(seed=42)
        env = HFTEnv(sim, max_steps=100)
        finetuner = FlowRLFinetuner(
            meanflow_model=meanflow,
            noise_policy=noise_policy,
            env=env,
            learning_rate=1e-3,
        )
        assert finetuner is not None


class TestSubConfigs:
    def test_market_config(self):
        mc = MarketConfig(sigma=0.5)
        assert mc.sigma == 0.5
        assert mc.S0 == 100.0

    def test_model_config(self):
        mc = ModelConfig(hidden_dim=256, T_pred=16)
        assert mc.hidden_dim == 256
        assert mc.T_pred == 16

    def test_pretrain_config(self):
        pc = PretrainConfig(num_epochs=50, batch_size=128)
        assert pc.num_epochs == 50
        assert pc.batch_size == 128

    def test_finetune_config(self):
        fc = FinetuneConfig(clip_epsilon=0.1, rollout_steps=4096)
        assert fc.clip_epsilon == 0.1
        assert fc.rollout_steps == 4096
