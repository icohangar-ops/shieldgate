"""Tests for neural network models (FiLM, MeanFlow, NoisePolicy)."""

import pytest
import torch
import torch.nn as nn

from finflowrl.models import FiLMLayer, MeanFlowPolicy, NoisePolicy


class TestFiLMLayer:
    def test_creation(self):
        film = FiLMLayer(state_dim=10, feat_dim=20)
        assert hasattr(film, "gamma")
        assert hasattr(film, "beta")

    def test_forward(self):
        film = FiLMLayer(state_dim=10, feat_dim=20)
        h = torch.randn(4, 20)
        s = torch.randn(4, 10)
        out = film(h, s)
        assert out.shape == (4, 20)

    def test_initial_identity(self):
        """Near-identity init: h' ≈ h when s ≈ 0."""
        film = FiLMLayer(state_dim=10, feat_dim=20)
        h = torch.randn(4, 20)
        s = torch.zeros(4, 10)
        out = film(h, s)
        assert torch.allclose(out, h, atol=0.1)

    def test_conditioning_changes_output_after_weight_perturbation(self):
        """FiLM starts as near-identity; after perturbing weights, it should differ."""
        film = FiLMLayer(state_dim=10, feat_dim=20)
        h = torch.randn(4, 20)
        s1 = torch.randn(4, 10)
        s2 = torch.randn(4, 10)
        # At init, weights are zero so all conditioning signals produce same output
        out_init = film(h, s1)
        assert torch.allclose(out_init, h, atol=1e-5), "Should be identity at init"
        # After perturbing weights, conditioning should have effect
        with torch.no_grad():
            nn.init.normal_(film.gamma.weight, std=1.0)
            nn.init.normal_(film.beta.weight, std=1.0)
        out1 = film(h, s1)
        out2 = film(h, s2)
        diff = (out1 - out2).abs().mean()
        assert diff > 0.01, f"FiLM conditioning has minimal effect after perturbation: diff={diff}"


class TestMeanFlowPolicy:
    def _make_model(self, **kwargs):
        defaults = dict(
            state_dim=14, action_dim=2, noise_dim=8,
            T_obs=2, T_pred=4, T_exec=2,
            hidden_dim=32, num_layers=2,
        )
        defaults.update(kwargs)
        return MeanFlowPolicy(**defaults)

    def test_creation(self):
        model = self._make_model()
        assert model.state_dim == 14
        assert model.T_pred == 4
        assert model.T_exec == 2
        assert model.chunk_noise_dim == 4 * 8  # 32

    def test_parameter_count(self):
        model = self._make_model()
        assert model.count_parameters() > 0

    def test_forward_training(self):
        model = self._make_model()
        B = 4
        states = torch.randn(B, 2, 14)
        actions = torch.rand(B, 4, 2) * 1.99 + 0.01
        loss, info = model(states, actions)
        assert torch.isfinite(loss)
        assert isinstance(info, dict)
        assert "loss" in info

    def test_backward(self):
        model = self._make_model()
        states = torch.randn(4, 2, 14)
        actions = torch.rand(4, 4, 2) * 1.99 + 0.01
        loss, _ = model(states, actions)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad

    def test_generate(self):
        model = self._make_model()
        model.eval()
        states = torch.randn(4, 2, 14)
        with torch.no_grad():
            out = model.generate(states)
        assert out.shape == (4, 4, 2)
        assert out.min() >= 0.01 - 1e-6
        assert out.max() <= 2.0 + 1e-6

    def test_act(self):
        model = self._make_model()
        model.eval()
        states = torch.randn(4, 2, 14)
        with torch.no_grad():
            action = model.act(states)
        assert action.shape == (4, 2)

    def test_encode_decode_roundtrip(self):
        model = self._make_model()
        a_in = torch.rand(2, 4, 2)
        latent = model.encode_actions(a_in)
        assert latent.shape == (2, 32)
        a_out = model.decode_actions(latent)
        assert a_out.shape == (2, 4, 2)

    def test_film_conditioning_effect(self):
        model = self._make_model()
        model.eval()
        s1 = torch.randn(2, 2, 14)
        s2 = torch.randn(2, 2, 14) * 5.0
        with torch.no_grad():
            torch.manual_seed(99)
            a1 = model.generate(s1)
            torch.manual_seed(99)
            a2 = model.generate(s2)
        assert (a1 - a2).abs().mean() > 0

    def test_t_exec_clamped(self):
        model = MeanFlowPolicy(T_pred=4, T_exec=10)
        assert model.T_exec == 4


class TestNoisePolicy:
    def _make_policy(self, **kwargs):
        defaults = dict(state_dim=28, noise_dim=32, hidden_dim=32, num_layers=2)
        defaults.update(kwargs)
        return NoisePolicy(**defaults)

    def test_creation(self):
        policy = self._make_policy()
        assert policy.state_dim == 28
        assert policy.noise_dim == 32

    def test_forward(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        mu, std = policy(states)
        assert mu.shape == (4, 32)
        assert std.shape == (4, 32)
        assert (std > 0).all()

    def test_sample_stochastic(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        noise = policy.sample(states, deterministic=False)
        assert noise.shape == (4, 32)

    def test_sample_deterministic(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        noise_det = policy.sample(states, deterministic=True)
        mu, _ = policy(states)
        assert torch.allclose(noise_det, mu, atol=1e-6)

    def test_log_prob(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        noise = policy.sample(states)
        lp = policy.log_prob(noise, states)
        assert lp.shape == (4, 32)
        assert torch.isfinite(lp).all()

    def test_entropy(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        ent = policy.entropy(states)
        assert ent.shape == (4,)
        assert torch.isfinite(ent).all()
        assert torch.allclose(ent, ent[0:1].expand_as(ent))

    def test_gradient_flow(self):
        policy = self._make_policy()
        s = torch.randn(4, 28, requires_grad=True)
        mu, std = policy(s)
        loss = mu.sum() + std.sum()
        loss.backward()
        assert s.grad is not None
        assert policy.log_std.grad is not None

    def test_kl_divergence(self):
        policy = self._make_policy()
        old_mu = torch.zeros(4, 32)
        old_log_std = torch.zeros(32)
        kl = policy.kl_divergence(old_mu, old_log_std)
        assert kl.shape == (4,)
        assert torch.isfinite(kl).all()
        assert (kl >= -1e-6).all()

    def test_gradient_clipping(self):
        policy = self._make_policy()
        states = torch.randn(4, 28)
        noise = policy.sample(states)
        lp = policy.log_prob(noise, states).sum(dim=-1)
        loss = (lp * torch.randn(4)).mean()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        assert torch.isfinite(total_norm)
