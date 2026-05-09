"""Feature-wise Linear Modulation (FiLM) conditioning layer.

Implements the FiLM conditioning mechanism from Perez et al. (2018)
"FiLM: Visual Reasoning with a General Conditioning Layer", which
conditionally modulates hidden features via an affine transformation
parameterised by a context vector:

    h' = gamma(s) * h + beta(s)

This is used throughout FinFlowRL to condition the MeanFlow velocity
network on the encoded market state, allowing a single network to
produce market-context-dependent velocity predictions.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Conditionally modulates network hidden features based on a state /
    context vector.  Each feature dimension is independently scaled by
    ``gamma(s)`` and shifted by ``beta(s)``:

        h' = gamma(s) * h + beta(s)

    Parameters are initialised so that the layer acts as a near-identity
    at the start of training (``gamma ≈ 1``, ``beta ≈ 0``), which
    stabilises early optimisation.

    Parameters
    ----------
    state_dim : int
        Dimension of the conditioning state / context vector *s*.
    feat_dim : int
        Dimension of the features *h* to modulate.

    Shape
    -----
    - **h**: ``((*B), feat_dim)`` where *B* is an optional batch prefix.
    - **s**: ``((*B), state_dim)``
    - **output**: ``((*B), feat_dim)``

    Examples
    --------
    >>> film = FiLMLayer(state_dim=64, feat_dim=128)
    >>> h = torch.randn(32, 128)
    >>> s = torch.randn(32, 64)
    >>> h_mod = film(h, s)
    >>> h_mod.shape
    torch.Size([32, 128])
    """

    def __init__(self, state_dim: int, feat_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.feat_dim = feat_dim

        # Scale (gamma) and shift (beta) projections
        self.gamma = nn.Linear(state_dim, feat_dim)
        self.beta = nn.Linear(state_dim, feat_dim)

        # Near-identity initialisation: gamma(s=0) ≈ 1, beta(s=0) ≈ 0
        # gamma(0) = W @ 0 + b = b  ⟹  b_gamma = ones
        # beta(0)  = W @ 0 + b = b  ⟹  b_beta  = zeros
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        h : torch.Tensor
            Input features to modulate, shape ``(*B, feat_dim)``.
        s : torch.Tensor
            Conditioning state vector, shape ``(*B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Modulated features, shape ``(*B, feat_dim)``.
        """
        gamma = self.gamma(s)   # (*B, feat_dim)
        beta = self.beta(s)     # (*B, feat_dim)
        return gamma * h + beta

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, "
            f"feat_dim={self.feat_dim}"
        )


# ======================================================================
# Quick smoke test
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Test 1: Basic forward pass
    film = FiLMLayer(state_dim=64, feat_dim=128)
    h = torch.randn(32, 128)
    s = torch.randn(32, 64)
    h_out = film(h, s)
    assert h_out.shape == (32, 128), f"Expected (32, 128), got {h_out.shape}"
    print(f"[PASS] Basic forward pass: output shape = {h_out.shape}")

    # Test 2: Near-identity at init (gamma≈1, beta≈0)
    with torch.no_grad():
        s_zero = torch.zeros(1, 64)
        h_test = torch.randn(1, 128)
        h_out_zero = film(h_test, s_zero)
        # With zero state: gamma(s=0) ≈ 1, beta(s=0) ≈ 0
        # So output ≈ input
        diff = (h_out_zero - h_test).abs().max().item()
        assert diff < 1e-5, f"Near-identity check failed: max diff = {diff}"
    print(f"[PASS] Near-identity at init: max deviation = {diff:.2e}")

    # Test 3: Different batch sizes and leading dimensions
    film2 = FiLMLayer(state_dim=14, feat_dim=256)
    h3d = torch.randn(4, 8, 256)
    s3d = torch.randn(4, 8, 14)
    h3d_out = film2(h3d, s3d)
    assert h3d_out.shape == (4, 8, 256), f"Expected (4, 8, 256), got {h3d_out.shape}"
    print(f"[PASS] 3-D input: output shape = {h3d_out.shape}")

    # Test 4: Gradient flow
    h_grad = torch.randn(8, 128, requires_grad=True)
    s_grad = torch.randn(8, 64, requires_grad=True)
    y = film(h_grad, s_grad).sum()
    y.backward()
    assert h_grad.grad is not None and h_grad.grad.shape == h_grad.shape
    assert s_grad.grad is not None and s_grad.grad.shape == s_grad.shape
    print("[PASS] Gradient flow: both h and s receive gradients")

    # Test 5: repr
    print(f"  repr: {film}")
    print("\nfilm.py OK – all tests passed.")
