"""MeanFlow policy for financial stochastic control.

Implements the MeanFlow matching framework (Geng et al. 2025) adapted for
financial market-making action generation.  The model is trained in two
stages:

**Stage 1 – Pre-training (Flow Matching on Expert Demonstrations)**
    Learns a conditional average-velocity field *u_θ(z_t, r, t, s)* by
    matching the MeanFlow consistency identity on straight-line
    interpolants between Gaussian noise and expert actions.

**Stage 2 – Fine-tuning (FlowRL)**
    A lightweight :class:`NoisePolicy` is trained via PPO to produce
    noise offsets, while the MeanFlow policy is frozen.  See
    ``noise_policy.py`` for details.

MeanFlow Formulation
--------------------
Instead of modelling the instantaneous velocity *v(z_t, t)*, MeanFlow
models the **average** velocity between two time-steps:

    u(z_t, r, t) = (1 / (t - r)) ∫_r^t v(z_τ, τ) dτ

The MeanFlow identity relates the average to the instantaneous velocity:

    u(z_t, r, t) = v(z_t, t) − (t − r) · d_t[u(z_t, r, t)]

Training Loss:

    L(θ) = E_{t,r,z_t,s} [ ‖u_θ + (t−r)·d_t[u_θ] − v_t‖² ]

where ``v_t = a_expert − z_0`` is the straight-line velocity.

One-Step Generation:

    a = z_1 − u_θ(z_1, 0, 1, s)

Action Chunking
---------------
Rather than generating a single action, the model predicts a sequence
of ``T_pred`` actions, executes the first ``T_exec`` actions, then
re-plans.  This non-Markovian formulation captures temporal dependencies
in market-making dynamics.

References
----------
    - FinFlowRL (arXiv 2509.17964)
    - MeanFlow: Training Straight Normalizing Flows using Jacobian
      Regularization (Geng et al. 2025)
    - Perez et al., "FiLM: Visual Reasoning with a General
      Conditioning Layer" (AAAI 2018)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLMLayer


# ======================================================================
# Helper: build a FiLM-conditioned MLP block
# ======================================================================

class _FilmMLPBlock(nn.Module):
    """Single residual MLP block with FiLM conditioning.

    Architecture: Linear → FiLM(state) → SiLU → Linear → residual
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        state_dim: int,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.film = FiLMLayer(state_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()
        self.residual: Optional[nn.Linear] = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        )

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Hidden state, shape ``(*B, in_dim)``.
        s : torch.Tensor
            Conditioning state, shape ``(*B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Output, shape ``(*B, out_dim)``.
        """
        residual = h if self.residual is None else self.residual(h)
        h = self.act(self.film(self.linear1(h), s))
        h = self.linear2(h)
        return h + residual


# ======================================================================
# MeanFlow Policy
# ======================================================================

class MeanFlowPolicy(nn.Module):
    """MeanFlow policy for financial stochastic control.

    A flow-matching model that learns the average velocity field for
    one-step action generation from market observations.  Supports
    **action chunking** (predicting ``T_pred`` actions per forward pass)
    and **FiLM conditioning** on the encoded market state.

    The flow operates in a latent noise space of dimension
    ``noise_dim`` per timestep.  Expert actions are encoded into this
    space via a lightweight encoder, and the generated latent is decoded
    back to action space for execution.

    Parameters
    ----------
    state_dim : int
        Dimension of a single market observation vector.
        Default ``14`` (mid-price, inventory, spread, intensities, …).
    action_dim : int
        Dimension of a single action (e.g. ``2`` for
        ``(delta_bid, delta_ask)``).  Default ``2``.
    noise_dim : int
        Per-timestep latent noise dimension for the flow.  Default ``16``.
        For fastest training set equal to ``action_dim``; larger values
        give the model more expressiveness at the cost of training speed.
    T_obs : int
        Number of past observations stacked as input.  Default ``2``.
    T_pred : int
        Number of future actions to predict (chunk size).  Default ``8``.
    T_exec : int
        Number of predicted actions to execute before re-planning.
        Default ``4``.  Must be ``<= T_pred``.
    hidden_dim : int
        Hidden dimension for all MLP layers.  Default ``128``.
    num_layers : int
        Number of FiLM-conditioned residual blocks.  Default ``3``.
    action_min : float
        Minimum clamp value for generated actions.  Default ``0.01``.
    action_max : float
        Maximum clamp value for generated actions.  Default ``2.0``.

    Notes
    -----
    * ``T_exec <= T_pred`` is enforced in ``__init__``.
    * The total latent dimension is ``chunk_noise_dim = T_pred * noise_dim``.
    * The MeanFlow loss requires second-order gradients
      (``create_graph=True``) and therefore needs slightly more memory
      than a standard flow-matching loss.
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 2,
        noise_dim: int = 16,
        T_obs: int = 2,
        T_pred: int = 8,
        T_exec: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        action_min: float = 0.01,
        action_max: float = 2.0,
    ):
        super().__init__()

        # Store configuration
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_exec = min(T_exec, T_pred)  # safety clamp
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_min = action_min
        self.action_max = action_max
        self.chunk_noise_dim = T_pred * noise_dim  # total latent dim

        # ----------------------------------------------------------
        # State encoder: (T_obs * state_dim) → hidden_dim
        # ----------------------------------------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(T_obs * state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ----------------------------------------------------------
        # Time encoder: (t, r, t-r) → hidden_dim
        # ----------------------------------------------------------
        self.time_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----------------------------------------------------------
        # Noise projection: chunk_noise_dim → hidden_dim
        # ----------------------------------------------------------
        self.noise_proj = nn.Sequential(
            nn.Linear(self.chunk_noise_dim, hidden_dim),
            nn.SiLU(),
        )

        # ----------------------------------------------------------
        # Velocity backbone: FiLM-conditioned residual MLP
        # Input: projected noise + time features
        # ----------------------------------------------------------
        in_dim = hidden_dim * 2  # noise_proj + time_feat
        self.vel_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.vel_blocks.append(
                _FilmMLPBlock(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim,  # state_feat dim (FiLM conditioning)
                )
            )

        # ----------------------------------------------------------
        # Velocity head: hidden_dim → chunk_noise_dim
        # ----------------------------------------------------------
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_noise_dim),
        )

        # ----------------------------------------------------------
        # Action encoder: action_dim → noise_dim  (per-timestep)
        # Shared across timesteps for parameter efficiency.
        # ----------------------------------------------------------
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, noise_dim),
        )

        # ----------------------------------------------------------
        # Action decoder: noise_dim → action_dim  (per-timestep)
        # Shared across timesteps.  Final tanh maps to [-1, 1] then
        # rescaled to [action_min, action_max].
        # ----------------------------------------------------------
        self.action_decoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        # Pre-compute action rescaling constants
        self.register_buffer(
            "_action_scale",
            torch.tensor((action_max - action_min) / 2.0),
        )
        self.register_buffer(
            "_action_bias",
            torch.tensor((action_max + action_min) / 2.0),
        )

        # ----------------------------------------------------------
        # Initialise weights
        # ----------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming initialisation to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, FiLMLayer):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def encode_state(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a window of ``T_obs`` market observations.

        Parameters
        ----------
        states : torch.Tensor
            Market observations.

            - Shape ``(B, T_obs, state_dim)`` — standard input.
            - Shape ``(B, T_obs * state_dim)`` — pre-flattened.

        Returns
        -------
        torch.Tensor
            Encoded state features, shape ``(B, hidden_dim)``.
        """
        if states.dim() == 3:
            states = states.reshape(states.size(0), -1)
        return self.state_encoder(states)

    def encode_time(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Encode flow time parameters ``(t, r, t-r)``.

        Parameters
        ----------
        t : torch.Tensor
            Current flow time, shape ``(B, 1)`` or broadcastable.
        r : torch.Tensor
            Reference time, shape ``(B, 1)`` or broadcastable.

        Returns
        -------
        torch.Tensor
            Time features, shape ``(B, hidden_dim)``.
        """
        time_feat = torch.cat([t, r, t - r], dim=-1)
        return self.time_encoder(time_feat)

    def encode_actions(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode expert actions into latent noise space.

        Maps each timestep's action from ``action_dim`` to ``noise_dim``
        using the shared encoder, then flattens to ``chunk_noise_dim``.

        Parameters
        ----------
        actions : torch.Tensor
            Expert actions, shape ``(B, T_pred, action_dim)``.

        Returns
        -------
        torch.Tensor
            Latent actions, shape ``(B, T_pred * noise_dim)``.
        """
        B = actions.size(0)
        # Reshape to (B * T_pred, action_dim) for batch processing
        a_flat = actions.reshape(B * self.T_pred, self.action_dim)
        # Encode each timestep
        z_flat = self.action_encoder(a_flat)  # (B * T_pred, noise_dim)
        # Reshape back to (B, T_pred * noise_dim)
        return z_flat.reshape(B, self.chunk_noise_dim)

    def decode_actions(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent noise into action space.

        Splits the latent vector into ``T_pred`` chunks of ``noise_dim``,
        decodes each to ``action_dim``, then rescales via tanh to
        ``[action_min, action_max]``.

        Parameters
        ----------
        latent : torch.Tensor
            Latent action sequence, shape ``(B, T_pred * noise_dim)``.

        Returns
        -------
        torch.Tensor
            Clamped actions, shape ``(B, T_pred, action_dim)``, values
            in ``[action_min, action_max]``.
        """
        B = latent.size(0)
        # Reshape to (B * T_pred, noise_dim)
        z_flat = latent.reshape(B * self.T_pred, self.noise_dim)
        # Decode: (B * T_pred, action_dim), values in [-1, 1]
        a_flat = self.action_decoder(z_flat)
        # Rescale from [-1, 1] → [action_min, action_max]
        a_flat = a_flat * self._action_scale + self._action_bias
        # Reshape to (B, T_pred, action_dim)
        actions = a_flat.reshape(B, self.T_pred, self.action_dim)
        # Final safety clamp
        return actions.clamp(self.action_min, self.action_max)

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the MeanFlow average velocity *u_θ(z_t, r, t, s)*.

        Parameters
        ----------
        z_t : torch.Tensor
            Interpolated latent point, shape ``(B, chunk_noise_dim)``.
            Must have ``requires_grad = True`` if computing the
            MeanFlow loss (needed for Jacobian computation).
        t : torch.Tensor
            Current flow time, shape ``(B, 1)``.
        r : torch.Tensor
            Reference time, shape ``(B, 1)``.
        state_feat : torch.Tensor
            Encoded market state, shape ``(B, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted average velocity, shape ``(B, chunk_noise_dim)``.
        """
        # Project noise to hidden dim
        h = self.noise_proj(z_t)  # (B, hidden_dim)

        # Encode time
        t_feat = self.encode_time(t, r)  # (B, hidden_dim)

        # Concatenate noise projection + time features
        h = torch.cat([h, t_feat], dim=-1)  # (B, 2*hidden_dim)

        # Pass through FiLM-conditioned residual blocks
        for block in self.vel_blocks:
            h = block(h, state_feat)  # (B, hidden_dim)

        # Velocity head
        velocity = self.vel_head(h)  # (B, chunk_noise_dim)
        return velocity

    # ------------------------------------------------------------------
    # MeanFlow consistency loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        u_pred: torch.Tensor,
        z_t: torch.Tensor,
        a_latent: torch.Tensor,
        z_0: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the MeanFlow consistency training loss.

        The MeanFlow identity states:

            u(z_t, r, t) + (t − r) · d_t[u(z_t, r, t)] = v_t

        where ``v_t = a_latent − z_0`` is the known straight-line
        velocity and ``d_t`` denotes the total time derivative:

            d_t[u] = ∂u/∂t + J_u · v_t

        The loss penalises deviations from this identity:

            L = ‖u_θ + (t−r)·d_t[u_θ] − v_t‖²

        The total derivative ``d_t[u_θ]`` is computed via ``torch.autograd``
        since ``z_t`` depends on ``t`` through the interpolation path.

        Parameters
        ----------
        u_pred : torch.Tensor
            Predicted velocity, shape ``(B, D)``.  Must be connected to
            the computational graph with ``requires_grad`` on ``t`` and
            ``z_t``.
        z_t : torch.Tensor
            Interpolated latent, shape ``(B, D)``.  Must be connected to
            ``t`` in the computational graph.
        a_latent : torch.Tensor
            Expert actions in latent space, shape ``(B, D)``.
        z_0 : torch.Tensor
            Source noise, shape ``(B, D)``.
        t : torch.Tensor
            Flow time, shape ``(B, 1)``, ``requires_grad = True``.
        r : torch.Tensor
            Reference time, shape ``(B, 1)``.

        Returns
        -------
        torch.Tensor
            Scalar MeanFlow consistency loss.

        Notes
        -----
        This method requires ``create_graph = True`` in the backward
        pass (handled internally), enabling gradient flow through the
        Jacobian terms.  Memory usage is higher than a standard
        flow-matching loss.
        """
        B, D = u_pred.shape
        v_t = a_latent - z_0  # (B, D) straight-line velocity
        dt = (t - r).clamp(min=1e-8)  # (B, 1) avoid division issues

        # ------------------------------------------------------------------
        # Compute total time derivative  d_t[u_θ(z_t(t), t)]
        #
        # Since z_t = (1-t)*z_0 + t*a_latent, autograd automatically
        # tracks the z_t → t dependency.  For each output component j:
        #
        #   d_t u_j = ∂u_j/∂t + (∂u_j/∂z_t) · (dz_t/dt)
        #           = ∂u_j/∂t + J_u[j,:] · v_t
        #
        # We compute per-component gradients via autograd.grad with
        # one-hot grad_outputs.
        # ------------------------------------------------------------------

        # Vector-Jacobian product: J_u @ v_t  (efficient single call)
        jvp_z = torch.autograd.grad(
            outputs=u_pred,
            inputs=z_t,
            grad_outputs=v_t,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, D)

        # ------------------------------------------------------------------
        # Compute the total time derivative per component.
        #
        # Since z_t = (1-t)*z_0 + t*a_latent depends on t, autograd
        # automatically tracks the total derivative:
        #
        #   d_t u_j = ∂u_j/∂t + (J_u[j,:] · v_t)
        #
        # We compute this per-component via one-hot grad_outputs.
        # ------------------------------------------------------------------
        d_t_u = torch.zeros_like(u_pred)  # (B, D)

        for j in range(D):
            # u_pred[:, j] has shape (B,) — grad_outputs must match
            d_t_j = torch.autograd.grad(
                outputs=u_pred[:, j],
                inputs=t,
                grad_outputs=torch.ones(B, device=u_pred.device, dtype=u_pred.dtype),
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, 1)
            d_t_u[:, j : j + 1] = d_t_j

        # ------------------------------------------------------------------
        # Consistency error:  u_θ + (t−r)·d_t[u_θ] − v_t
        # ------------------------------------------------------------------
        consistency_error = u_pred + dt * d_t_u - v_t  # (B, D)

        # Mean squared error over batch and dimensions
        loss = (consistency_error ** 2).mean()
        return loss

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        a_expert: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Training forward pass: compute the MeanFlow consistency loss.

        Samples flow times ``t`` and reference times ``r``, constructs
        the straight-line interpolant, predicts the average velocity,
        and returns the MeanFlow consistency loss.

        Parameters
        ----------
        states : torch.Tensor
            Market observations, shape ``(B, T_obs, state_dim)``.
        a_expert : torch.Tensor
            Expert actions, shape ``(B, T_pred, action_dim)``.
        t : torch.Tensor or None
            Flow time, shape ``(B, 1)``.  If ``None``, sampled
            uniformly from ``[eps, 1]``.
        r : torch.Tensor or None
            Reference time, shape ``(B, 1)``.  If ``None``, sampled
            uniformly from ``[eps, t]``.

        Returns
        -------
        loss : torch.Tensor
            Scalar MeanFlow consistency loss.
        info : dict
            Diagnostic information (loss components, shapes, etc.).
        """
        B = states.size(0)
        device = states.device
        dtype = states.dtype

        # Sample time parameters if not provided
        if t is None:
            t = torch.rand(B, 1, device=device, dtype=dtype)
            t = (t * 0.999 + 0.001).requires_grad_(True)
        else:
            t = t.detach().clone().requires_grad_(True)

        if r is None:
            r = torch.rand(B, 1, device=device, dtype=dtype) * t
            r = r.clamp(min=1e-4)
        else:
            r = r.detach()

        # Sample source noise
        z_0 = torch.randn(
            B, self.chunk_noise_dim, device=device, dtype=dtype
        )

        # Encode expert actions to latent space
        a_latent = self.encode_actions(a_expert)  # (B, chunk_noise_dim)

        # Straight-line interpolant: z_t = (1-t)*z_0 + t*a_latent
        # z_t depends on t, which is required for MeanFlow loss gradients
        z_t = (1 - t) * z_0 + t * a_latent

        # Encode market state (detach to keep gradient flow clean)
        state_feat = self.encode_state(states)

        # Predict average velocity
        u_pred = self.predict_velocity(z_t, t, r, state_feat)

        # Compute MeanFlow consistency loss
        loss = self.compute_loss(u_pred, z_t, a_latent, z_0, t, r)

        # Diagnostic info
        with torch.no_grad():
            v_t = a_latent - z_0
            info = {
                "loss": loss.item(),
                "t_mean": t.mean().item(),
                "r_mean": r.mean().item(),
                "dt_mean": (t - r).mean().item(),
                "u_pred_norm": u_pred.norm(dim=-1).mean().item(),
                "v_t_norm": v_t.norm(dim=-1).mean().item(),
                "z_t_norm": z_t.norm(dim=-1).mean().item(),
            }

        return loss, info

    # ------------------------------------------------------------------
    # One-step generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        states: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Generate actions via one-step MeanFlow denoising.

        Implements the one-step generation formula:

            a_latent = z_1 − u_θ(z_1, 0, 1, s)
            a = decode(a_latent)

        where ``z_1 ~ N(0, I)`` and ``decode`` maps latent to the
        clamped action space ``[action_min, action_max]``.

        Parameters
        ----------
        states : torch.Tensor
            Market observations, shape ``(B, T_obs, state_dim)``.
        deterministic : bool
            If ``True`` (default), use a fixed random seed per call
            for reproducibility during evaluation.  Set to ``False``
            for stochastic exploration during deployment.

        Returns
        -------
        torch.Tensor
            Generated action chunk, shape ``(B, T_pred, action_dim)``,
            values clamped to ``[action_min, action_max]``.
        """
        B = states.size(0)
        device = states.device

        # Sample endpoint noise z_1 ~ N(0, I)
        z_1 = torch.randn(
            B, self.chunk_noise_dim, device=device, dtype=states.dtype
        )

        # Fixed time parameters for one-step generation
        t = torch.ones(B, 1, device=device, dtype=states.dtype)
        r = torch.zeros(B, 1, device=device, dtype=states.dtype)

        # Encode state
        state_feat = self.encode_state(states)

        # Predict average velocity at (z_1, 0, 1, s)
        u = self.predict_velocity(z_1, t, r, state_feat)

        # One-step denoising: a_latent = z_1 − u_θ(z_1, 0, 1, s)
        a_latent = z_1 - u

        # Decode from latent space to action space
        actions = self.decode_actions(a_latent)

        return actions

    @torch.no_grad()
    def act(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a single action (first of the chunk).

        Convenience method that calls :meth:`generate` and returns only
        the action at the current timestep (index 0 of the chunk).

        Parameters
        ----------
        states : torch.Tensor
            Market observations, shape ``(B, T_obs, state_dim)``.

        Returns
        -------
        torch.Tensor
            Action for the current timestep, shape ``(B, action_dim)``.
        """
        chunk = self.generate(states)
        return chunk[:, 0, :]  # (B, action_dim)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"noise_dim={self.noise_dim}, "
            f"T_obs={self.T_obs}, T_pred={self.T_pred}, T_exec={self.T_exec}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"chunk_noise_dim={self.chunk_noise_dim}"
        )


# ======================================================================
# Quick smoke test
# ======================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Test 1: Model creation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 1: Model creation")
    model = MeanFlowPolicy()
    print(f"  {model}")
    n_params = model.count_parameters()
    print(f"  Trainable parameters: {n_params:,}")
    assert n_params > 0, "No trainable parameters!"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 2: Forward pass (training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Forward pass (training)")
    B = 16
    states = torch.randn(B, model.T_obs, model.state_dim)
    a_expert = torch.rand(B, model.T_pred, model.action_dim)  # in [0, 1]
    # Rescale to realistic action range
    a_expert = a_expert * (model.action_max - model.action_min) + model.action_min

    loss, info = model(states, a_expert)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Info: {info}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 3: Backward pass (gradient flow)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: Backward pass (gradient flow)")
    loss, _ = model(states, a_expert)
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
        else:
            grad_norms.append((name, 0.0))

    has_grads = sum(1 for _, g in grad_norms if g > 0)
    no_grads = sum(1 for _, g in grad_norms if g == 0.0)
    print(f"  Parameters with gradients: {has_grads}")
    print(f"  Parameters without gradients: {no_grads}")
    assert has_grads > 0, "No gradients received!"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 4: Generation (one-step)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 4: One-step generation")
    model.eval()
    with torch.no_grad():
        actions = model.generate(states)
    print(f"  Output shape: {actions.shape}")
    print(f"  Expected shape: ({B}, {model.T_pred}, {model.action_dim})")
    assert actions.shape == (B, model.T_pred, model.action_dim), (
        f"Shape mismatch: {actions.shape} vs "
        f"({B}, {model.T_pred}, {model.action_dim})"
    )
    print(f"  Action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    assert actions.min() >= model.action_min - 1e-6, (
        f"Action below minimum: {actions.min().item()}"
    )
    assert actions.max() <= model.action_max + 1e-6, (
        f"Action above maximum: {actions.max().item()}"
    )
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 5: Single action (act method)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 5: Single action via act()")
    single_action = model.act(states)
    print(f"  Output shape: {single_action.shape}")
    assert single_action.shape == (B, model.action_dim), (
        f"Shape mismatch: {single_action.shape}"
    )
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 6: Action encoder/decoder roundtrip
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 6: Action encoder/decoder roundtrip")
    model.train()
    a_in = torch.rand(4, model.T_pred, model.action_dim)
    a_latent = model.encode_actions(a_in)
    print(f"  Encoded shape: {a_latent.shape}")
    assert a_latent.shape == (4, model.chunk_noise_dim)
    a_out = model.decode_actions(a_latent)
    print(f"  Decoded shape: {a_out.shape}")
    assert a_out.shape == (4, model.T_pred, model.action_dim)
    print(f"  Decoded range: [{a_out.min().item():.4f}, {a_out.max().item():.4f}]")
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 7: FiLM conditioning effect
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 7: FiLM conditioning effect")
    model.eval()
    with torch.no_grad():
        states_a = torch.randn(4, model.T_obs, model.state_dim)
        states_b = torch.randn(4, model.T_obs, model.state_dim) * 5.0
        # Same noise seed for both
        torch.manual_seed(99)
        actions_a = model.generate(states_a)
        torch.manual_seed(99)
        actions_b = model.generate(states_b)
        diff = (actions_a - actions_b).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    assert diff > 0, "FiLM conditioning has no effect!"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 8: Different configuration (small model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 8: Small model configuration")
    small = MeanFlowPolicy(
        state_dim=8,
        action_dim=2,
        noise_dim=2,  # Same as action_dim for efficiency
        T_obs=1,
        T_pred=4,
        T_exec=2,
        hidden_dim=32,
        num_layers=2,
    )
    print(f"  {small}")
    print(f"  Parameters: {small.count_parameters():,}")
    s_small = torch.randn(4, 1, 8)
    a_small = torch.rand(4, 4, 2) * 1.99 + 0.01
    loss_s, info_s = small(s_small, a_small)
    print(f"  Loss: {loss_s.item():.6f}")
    assert torch.isfinite(loss_s), f"Small model loss not finite: {loss_s.item()}"
    loss_s.backward()
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("meanflow.py OK – all tests passed.")
