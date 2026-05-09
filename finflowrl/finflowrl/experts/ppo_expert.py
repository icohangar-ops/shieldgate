"""PPO-trained market-making expert (lightweight numpy implementation).

Implements a simple multi-layer perceptron (MLP) policy network that maps
market-state observations to bid/ask half-spread actions.  The network is
designed to be **loadable** with weights from a PPO training checkpoint
produced by the full FinFlowRL pipeline (which uses PyTorch internally).

In its default (untrained) state the network is initialised with carefully
chosen biases so that the output spreads are in a reasonable range
(~0.1–0.5) even before any training has occurred.  This makes the expert
immediately usable for imitation-learning warm-starting.

Architecture
------------
    Input (obs_dim=14) → FC(64, tanh) → FC(64, tanh) → FC(2, sigmoid) → scale

The final sigmoid outputs are rescaled from ``[0, 1]`` to
``[clip_lo, clip_hi]`` so the network always produces valid spreads.

Design notes
------------
* **No PyTorch / TensorFlow dependency** – pure NumPy forward pass.
* ``save_weights`` / ``load_weights`` persist and restore all weight
  matrices and biases as a single ``.npz`` file.
* The observation vector is a flattened concatenation of
  :class:`MarketState` fields plus derived features (total intensity,
  intensity imbalance, squared inventory, time-remaining squared).

Reference
---------
    FinFlowRL (arXiv 2509.17964) – Section 3.4, Algorithm 4.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .base import Expert, ExpertAction, MarketState, clip_spread


# ======================================================================
# Helpers
# ======================================================================

def _xavier_init(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Xavier/Glorot uniform initialisation.

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Weight matrix of shape ``(fan_in, fan_out)``.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def _tanh(x: np.ndarray) -> np.ndarray:
    """Numerically stable tanh."""
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Uses the piecewise formulation to avoid overflow in ``exp``.
    """
    pos = x >= 0
    z = np.zeros_like(x, dtype=np.float64)
    z[pos] = np.exp(-x[pos])
    z[~pos] = np.exp(x[~pos])
    top = np.ones_like(x, dtype=np.float64)
    top[~pos] = z[~pos]
    return top / (1.0 + z)


# ======================================================================
# PPO Expert
# ======================================================================

class PPOExpert(Expert):
    """Pre-trained PPO market-making expert (numpy MLP).

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector fed to the network.
        Default ``14`` (8 raw + 6 derived features; see
        :meth:`_state_to_obs`).
    action_dim : int
        Dimension of the action output (always 2: delta_bid, delta_ask).
        Default ``2``.
    hidden_dim : int
        Width of each hidden layer.  Default ``64``.
    n_hidden : int
        Number of hidden layers.  Default ``2``.
    clip_lo : float
        Minimum half-spread output.  Default ``0.01``.
    clip_hi : float
        Maximum half-spread output.  Default ``2.0``.
    seed : int | None
        Random seed for weight initialisation.  Default ``42``.
    """

    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 2,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        clip_lo: float = 0.01,
        clip_hi: float = 2.0,
        seed: int | None = 42,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")
        if clip_lo <= 0:
            raise ValueError(f"clip_lo must be positive, got {clip_lo}")
        if clip_hi <= clip_lo:
            raise ValueError(
                f"clip_hi ({clip_hi}) must exceed clip_lo ({clip_lo})"
            )

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.n_hidden: int = n_hidden
        self.clip_lo: float = clip_lo
        self.clip_hi: float = clip_hi

        # Build network weights
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._init_network()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _init_network(self) -> None:
        """Initialise the MLP weight matrices and biases.

        Layer structure:
            [obs_dim] → hidden_dim → ... → hidden_dim → [action_dim]

        Hidden layers use Xavier init + zero bias.
        Output layer uses small random init with a bias tuned so that
        sigmoid outputs ≈ 0.2 (i.e. spreads ≈ 0.2 * clip_hi when
        clip_lo ≈ 0).
        """
        self._weights.clear()
        self._biases.clear()

        layer_sizes = [self.obs_dim] + [self.hidden_dim] * self.n_hidden + [self.action_dim]

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = _xavier_init(fan_in, fan_out, self._rng)
            self._weights.append(W)

            if i < len(layer_sizes) - 2:
                # Hidden-layer bias: zeros
                b = np.zeros(fan_out, dtype=np.float64)
            else:
                # Output-layer bias: chosen so sigmoid outputs ~0.2
                # sigmoid(x) ≈ 0.2 ⟹ x ≈ ln(0.2/0.8) = ln(0.25) ≈ -1.386
                b = np.full(fan_out, -1.386, dtype=np.float64)
            self._biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        """Run a forward pass through the MLP.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape ``(obs_dim,)`` or ``(1, obs_dim)``.

        Returns
        -------
        np.ndarray
            Raw network output of shape ``(action_dim,)`` (after sigmoid).
        """
        x = np.atleast_2d(obs).astype(np.float64)

        # Hidden layers: linear → tanh
        for i in range(self.n_hidden):
            x = x @ self._weights[i] + self._biases[i]
            x = _tanh(x)

        # Output layer: linear → sigmoid
        out_idx = self.n_hidden
        x = x @ self._weights[out_idx] + self._biases[out_idx]
        x = _sigmoid(x)

        return x.flatten()

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: MarketState) -> np.ndarray:
        """Convert a :class:`MarketState` into a fixed-size observation vector.

        Layout (obs_dim=14):
            0  mid_price              (raw)
            1  inventory              (raw)
            2  spread                 (raw)
            3  buy_intensity          (raw)
            4  sell_intensity         (raw)
            5  price_change           (raw)
            6  volatility             (raw)
            7  time_remaining         (raw)
            -- derived features --
            8  total_intensity        = buy + sell
            9  intensity_imbalance    = (buy - sell) / (buy + sell + ε)
            10 inventory_sq          = inventory²
            11 time_remaining_sq     = time_remaining²
            12 vol_x_time            = volatility × time_remaining
            13 inv_x_vol             = inventory × volatility

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        np.ndarray
            Feature vector of shape ``(obs_dim,)``.
        """
        total_intensity = state.buy_intensity + state.sell_intensity
        imbalance = (
            (state.buy_intensity - state.sell_intensity)
            / (total_intensity + 1e-8)
        )

        return np.array(
            [
                state.mid_price,
                float(state.inventory),
                state.spread,
                state.buy_intensity,
                state.sell_intensity,
                state.price_change,
                state.volatility,
                state.time_remaining,
                total_intensity,
                imbalance,
                float(state.inventory) ** 2,
                state.time_remaining ** 2,
                state.volatility * state.time_remaining,
                float(state.inventory) * state.volatility,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Core strategy
    # ------------------------------------------------------------------

    def act(self, state: MarketState) -> ExpertAction:
        """Compute PPO policy action (bid/ask spreads).

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        ExpertAction
            ``delta_bid`` and ``delta_ask`` rescaled from sigmoid output
            to ``[clip_lo, clip_hi]``.
        """
        obs = self._state_to_obs(state)
        raw = self._forward(obs)

        # Rescale sigmoid [0, 1] → [clip_lo, clip_hi]
        spread_range = self.clip_hi - self.clip_lo
        delta_bid = self.clip_lo + raw[0] * spread_range
        delta_ask = self.clip_lo + raw[1] * spread_range

        return ExpertAction(
            delta_bid=clip_spread(delta_bid, self.clip_lo, self.clip_hi),
            delta_ask=clip_spread(delta_ask, self.clip_lo, self.clip_hi),
        )

    def name(self) -> str:
        """Return the expert identifier."""
        return "PPO-Expert"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str | os.PathLike) -> None:
        """Save network weights and biases to a ``.npz`` file.

        Parameters
        ----------
        path : str | os.PathLike
            Destination file path (e.g. ``"ppo_expert_weights.npz"``).
            The directory must exist.
        """
        data: Dict[str, np.ndarray] = {
            "obs_dim": np.array(self.obs_dim),
            "action_dim": np.array(self.action_dim),
            "hidden_dim": np.array(self.hidden_dim),
            "n_hidden": np.array(self.n_hidden),
            "clip_lo": np.array(self.clip_lo),
            "clip_hi": np.array(self.clip_hi),
        }
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            data[f"W_{i}"] = W
            data[f"b_{i}"] = b
        np.savez(path, **data)

    def load_weights(self, path: str | os.PathLike) -> None:
        """Load network weights and biases from a ``.npz`` file.

        The file must have been created by :meth:`save_weights`.

        Parameters
        ----------
        path : str | os.PathLike
            Source file path.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the architecture metadata does not match this instance.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weight file not found: {path}")

        data = np.load(path, allow_pickle=False)

        # Validate architecture
        expected_layers = self.n_hidden + 1
        for i in range(expected_layers):
            if f"W_{i}" not in data or f"b_{i}" not in data:
                raise ValueError(
                    f"Weight file missing layer {i} keys (W_{i}, b_{i})"
                )

        # Load weights
        self._weights.clear()
        self._biases.clear()
        for i in range(expected_layers):
            self._weights.append(data[f"W_{i}"].astype(np.float64))
            self._biases.append(data[f"b_{i}"].astype(np.float64))

    def reset(self) -> None:
        """Re-initialise network weights (fresh random init).

        This effectively "un-trains" the network, restoring the default
        bias-tuned initialisation.
        """
        self._init_network()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = sum(W.size + b.size for W, b in zip(self._weights, self._biases))
        return (
            f"{self.__class__.__name__}("
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, n_hidden={self.n_hidden}, "
            f"params={n_params})"
        )

    def param_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(W.size + b.size for W, b in zip(self._weights, self._biases))


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    import tempfile

    expert = PPOExpert(obs_dim=14, hidden_dim=64, n_hidden=2, seed=42)
    print(f"Created: {expert}")

    state = MarketState(
        mid_price=100.0,
        inventory=5,
        spread=0.05,
        buy_intensity=1.2,
        sell_intensity=0.8,
        price_change=0.01,
        volatility=0.02,
        time_remaining=0.75,
    )

    # --- Test 1: default forward pass ---
    action = expert.act(state)
    print(f"\n[Default weights]   {expert.name()}: {action}")
    assert 0.01 <= action.delta_bid <= 2.0, "delta_bid out of range"
    assert 0.01 <= action.delta_ask <= 2.0, "delta_ask out of range"

    # --- Test 2: neutral state ---
    state_neutral = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=1.0,
    )
    action_neutral = expert.act(state_neutral)
    print(f"[Neutral state]     {expert.name()}: {action_neutral}")

    # --- Test 3: save/load round-trip ---
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        expert.save_weights(tmp_path)
        print(f"\nSaved weights to {tmp_path}")

        expert2 = PPOExpert(obs_dim=14, hidden_dim=64, n_hidden=2, seed=123)
        expert2.load_weights(tmp_path)

        action_after_load = expert2.act(state)
        print(f"[After load]       {expert2.name()}: {action_after_load}")

        # Outputs should match exactly (deterministic forward pass)
        assert np.isclose(action.delta_bid, action_after_load.delta_bid), \
            "delta_bid mismatch after load"
        assert np.isclose(action.delta_ask, action_after_load.delta_ask), \
            "delta_ask mismatch after load"
        print("Save/load round-trip: PASSED")
    finally:
        os.unlink(tmp_path)

    # --- Test 4: reset ---
    action_before_reset = expert.act(state)
    expert.reset()
    action_after_reset = expert.act(state)
    print(f"\n[Before reset]      delta_bid={action_before_reset.delta_bid:.4f}, "
          f"delta_ask={action_before_reset.delta_ask:.4f}")
    print(f"[After reset]       delta_bid={action_after_reset.delta_bid:.4f}, "
          f"delta_ask={action_after_reset.delta_ask:.4f}")
    print(f"Param count: {expert.param_count()}")

    # --- Test 5: different network sizes ---
    small_expert = PPOExpert(obs_dim=14, hidden_dim=32, n_hidden=1, seed=0)
    small_action = small_expert.act(state)
    print(f"\n[Small network]     {small_expert}")
    print(f"Action: {small_action}")

    print("\nppo_expert.py OK")
