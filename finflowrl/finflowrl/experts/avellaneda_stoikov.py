"""Avellaneda-Stoikov market-making expert.

Implements the optimal quoting strategy from:

    Avellaneda, M. & Stoikov, S. (2008).  *High-frequency trading in a
    limit order book.*  Quantitative Finance, 8(3), 217–224.

The model derives a closed-form reservation price and symmetric half-spread
that maximises the agent's expected utility under exponential risk aversion
and Poisson order-arrival dynamics.

Key formulas
------------
Reservation price:

    r_t = S_t − q · γ · σ² · (T − t)

Symmetric half-spread:

    δ_t = (1/γ) · ln(1 + γ / κ)

Bid / Ask quotes:

    bid  = r_t − δ_t   →   delta_bid  = S_t − bid  = q·γ·σ²·(T−t) + δ_t
    ask  = r_t + δ_t   →   delta_ask  = ask − S_t  = δ_t − q·γ·σ²·(T−t)

where

    S_t   = current mid-price
    q     = signed inventory (positive = long)
    γ     = risk-aversion coefficient
    σ     = volatility estimate
    T − t = time remaining in the trading horizon
    κ     = order-arrival intensity parameter

Reference
---------
    FinFlowRL (arXiv 2509.17964) – Section 3.1, Algorithm 1.
"""

from __future__ import annotations

import numpy as np

from .base import Expert, ExpertAction, MarketState, clip_spread


class AvellanedaStoikovExpert(Expert):
    """Avellaneda-Stoikov optimal market-making strategy.

    Parameters
    ----------
    gamma : float
        Coefficient of exponential risk aversion (γ > 0).  Higher values
        make the agent more conservative, widening spreads.  Default ``0.1``.
    k : float
        Order-arrival intensity parameter (κ > 0).  Higher values mean
        more frequent fills, allowing tighter spreads.  Default ``1.5``.
    sigma : float
        Annualised volatility estimate (σ > 0).  If ``None``, the expert
        uses the volatility field from :class:`MarketState` directly.
        Default ``None``.
    T : float
        Full length of the trading horizon (e.g. 1.0 for one normalised
        period).  The effective time-to-expiry is ``T * time_remaining``
        where ``time_remaining`` is in ``[0, 1]``.  Default ``1.0``.
    clip_lo : float
        Minimum allowed half-spread.  Default ``0.01``.
    clip_hi : float
        Maximum allowed half-spread.  Default ``2.0``.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float | None = None,
        T: float = 1.0,
        clip_lo: float = 0.01,
        clip_hi: float = 2.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if k <= 0:
            raise ValueError(f"k (arrival intensity) must be positive, got {k}")
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if T <= 0:
            raise ValueError(f"T (horizon) must be positive, got {T}")

        self.gamma: float = gamma
        self.k: float = k
        self.sigma: float | None = sigma
        self.T: float = T
        self.clip_lo: float = clip_lo
        self.clip_hi: float = clip_hi

    # ------------------------------------------------------------------
    # Core strategy
    # ------------------------------------------------------------------

    def act(self, state: MarketState) -> ExpertAction:
        """Compute AS-optimal bid/ask half-spreads.

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        ExpertAction
            ``delta_bid`` and ``delta_ask`` clipped to ``[clip_lo, clip_hi]``.
        """
        # Use the volatility from MarketState unless overridden at init.
        sigma = self.sigma if self.sigma is not None else state.volatility
        sigma = max(sigma, 1e-8)  # guard against zero volatility

        # Effective time to expiry (T − t) in model units.
        tau = self.T * max(state.time_remaining, 0.0)

        # Inventory term: q · γ · σ² · (T − t)
        inventory_term = (
            state.inventory * self.gamma * (sigma ** 2) * tau
        )

        # Symmetric half-spread: (1/γ) · ln(1 + γ / κ)
        half_spread = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)

        # Bid / ask half-spreads from mid-price
        delta_bid = inventory_term + half_spread
        delta_ask = half_spread - inventory_term

        return ExpertAction(
            delta_bid=clip_spread(delta_bid, self.clip_lo, self.clip_hi),
            delta_ask=clip_spread(delta_ask, self.clip_lo, self.clip_hi),
        )

    def name(self) -> str:
        """Return the expert identifier."""
        return "Avellaneda-Stoikov"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, k={self.k}, "
            f"sigma={self.sigma}, T={self.T})"
        )


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    expert = AvellanedaStoikovExpert(gamma=0.1, k=1.5)

    # --- Test 1: neutral inventory ---
    state_neutral = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.2,
        sell_intensity=0.8,
        price_change=0.01,
        volatility=0.02,
        time_remaining=0.75,
    )
    action_neutral = expert.act(state_neutral)
    print(f"[Neutral inventory] {expert.name()}: {action_neutral}")

    # --- Test 2: long inventory (should widen bid, tighten ask) ---
    state_long = MarketState(
        mid_price=100.0,
        inventory=20,
        spread=0.05,
        buy_intensity=1.2,
        sell_intensity=0.8,
        price_change=0.01,
        volatility=0.02,
        time_remaining=0.5,
    )
    action_long = expert.act(state_long)
    print(f"[Long inventory]    {expert.name()}: {action_long}")

    # --- Test 3: short inventory (should tighten bid, widen ask) ---
    state_short = MarketState(
        mid_price=100.0,
        inventory=-15,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.5,
        price_change=-0.02,
        volatility=0.03,
        time_remaining=0.25,
    )
    action_short = expert.act(state_short)
    print(f"[Short inventory]   {expert.name()}: {action_short}")

    # --- Test 4: near expiry (inventory term vanishes) ---
    state_expiry = MarketState(
        mid_price=100.0,
        inventory=50,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=0.0,
    )
    action_expiry = expert.act(state_expiry)
    print(f"[Near expiry]       {expert.name()}: {action_expiry}")

    # --- Test 5: edge case – zero volatility fallback ---
    state_zero_vol = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.0,
        time_remaining=1.0,
    )
    action_zero_vol = expert.act(state_zero_vol)
    print(f"[Zero volatility]   {expert.name()}: {action_zero_vol}")

    print(f"\nRepr: {expert}")
    print("avellaneda_stoikov.py OK")
