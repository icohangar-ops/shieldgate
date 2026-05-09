"""Abstract base class for market-making expert strategies.

All expert strategies used in the FinFlowRL imitation-reinforcement learning
framework inherit from :class:`Expert`.  Each expert receives a
:class:`MarketState` snapshot and returns an :class:`ExpertAction` containing
the optimal bid/ask half-spreads around the mid-price.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ======================================================================
# Data containers
# ======================================================================

@dataclass
class ExpertAction:
    """Action returned by an expert strategy.

    Attributes
    ----------
    delta_bid : float
        Half-spread for the bid side, i.e. ``bid_price = mid_price - delta_bid``.
        Units are the same as the price (e.g. dollars, ticks).  Must be
        non-negative; typically clipped to ``[0.01, 2.0]`` by the environment.
    delta_ask : float
        Half-spread for the ask side, i.e. ``ask_price = mid_price + delta_ask``.
        Same units and constraints as ``delta_bid``.
    """

    delta_bid: float
    delta_ask: float


@dataclass
class MarketState:
    """Compact representation of the current market microstructure.

    This is the *observation* interface that every expert receives.  It is
    deliberately kept lightweight so that experts can be evaluated at high
    frequency without memory allocation overhead.

    Attributes
    ----------
    mid_price : float
        Current mid-price of the instrument.
    inventory : int
        Current inventory position (signed; positive = long).
    spread : float
        Current best bid-ask spread.
    buy_intensity : float
        Estimated intensity of aggressive buy orders (Poisson rate λ⁺).
    sell_intensity : float
        Estimated intensity of aggressive sell orders (Poisson rate λ⁻).
    price_change : float
        Most recent mid-price change (ΔS = S_t − S_{t−1}).
    volatility : float
        Current volatility estimate (e.g. EWMA of squared returns).
    time_remaining : float
        Fraction of the episode/trading horizon remaining, in ``[0, 1]``.
        Default ``1.0`` (full horizon ahead).
    """

    mid_price: float
    inventory: int
    spread: float
    buy_intensity: float
    sell_intensity: float
    price_change: float
    volatility: float
    time_remaining: float = 1.0


# ======================================================================
# Abstract base
# ======================================================================

class Expert(ABC):
    """Abstract base class for market-making expert strategies.

    Subclasses must implement :meth:`act` and :meth:`name`.  The optional
    :meth:`reset` hook allows experts to clear any internal state between
    episodes.

    Design notes
    ------------
    * Experts are **stateless by default** – they only depend on the
      :class:`MarketState` passed to :meth:`act`.  This makes them safe to
      share across parallel environment workers.
    * For experts that maintain internal state (e.g. a PPO neural network
      with hidden state), :meth:`reset` should clear that state.
    * The only allowed external dependency is **NumPy** – no PyTorch,
      TensorFlow, or JAX at this layer.
    """

    @abstractmethod
    def act(self, state: MarketState) -> ExpertAction:
        """Compute optimal bid/ask spreads given the current market state.

        Parameters
        ----------
        state : MarketState
            A snapshot of the current market microstructure.

        Returns
        -------
        ExpertAction
            The recommended bid and ask half-spreads.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this expert.

        Used for logging, dashboard labels, and imitation-learning
        weight labels.

        Returns
        -------
        str
            Expert identifier (e.g. ``"Avellaneda-Stoikov"``).
        """
        ...

    def reset(self) -> None:
        """Reset any internal state between episodes.

        The default implementation is a no-op.  Override in subclasses
        that maintain internal state (e.g. running averages, hidden
        activations).
        """
        pass


# ======================================================================
# Utility helpers
# ======================================================================

def clip_spread(spread: float, lo: float = 0.01, hi: float = 2.0) -> float:
    """Clamp a half-spread to a valid range.

    Parameters
    ----------
    spread : float
        Raw half-spread value.
    lo : float
        Minimum allowed spread (must be > 0).  Default ``0.01``.
    hi : float
        Maximum allowed spread.  Default ``2.0``.

    Returns
    -------
    float
        Clipped half-spread in ``[lo, hi]``.
    """
    return float(np.clip(spread, lo, hi))


# ======================================================================
# Quick smoke test
# ======================================================================

if __name__ == "__main__":
    # Create a dummy market state
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
    print(f"MarketState: {state}")
    print(f"Clip test:   clip_spread(-0.5) = {clip_spread(-0.5)}")
    print(f"Clip test:   clip_spread(5.0)  = {clip_spread(5.0)}")
    print(f"Clip test:   clip_spread(0.3)  = {clip_spread(0.3)}")
    print("base.py OK – all imports and helpers verified.")
