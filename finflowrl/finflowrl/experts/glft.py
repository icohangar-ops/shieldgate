"""Guéant-Lehalle-Fernandez-Tapia (GLFT) market-making expert.

Implements the closed-form approximation from:

    Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013).  *Dealing
    with the inventory risk: A solution to the market making problem.*
    Mathematics and Financial Economics, 7(4), 477–497.

The GLFT model extends Avellaneda-Stoikov by incorporating **asymmetric**
buy/sell order-arrival intensities (λ⁺, λ⁻) and providing a closed-form
approximation that avoids numerical root-finding.

Key formulas
------------
Base half-spreads (asymmetric):

    δ_b = (1/γ) · ln(1 + γ / κ_b)
    δ_a = (1/γ) · ln(1 + γ / κ_a)

where the intensity parameters absorb the Poisson arrival rates:

    κ_b = κ · λ⁻(t)          (bid-side: sells arrive)
    κ_a = κ · λ⁺(t)          (ask-side: buys arrive)

Inventory skew adjustment:

    η  = q · γ · σ² · (T − t) / 2

    δ_b ← δ_b + η     (widen bid when long → reluctant to buy more)
    δ_a ← δ_a − η     (widen ask when short → reluctant to sell more)

The asymmetry in κ means that in a one-sided market (e.g. heavy buy
intensity), the ask spread tightens (more fill probability) while the
bid spread widens (less urgency to replenish inventory).

Reference
---------
    FinFlowRL (arXiv 2509.17964) – Section 3.2, Algorithm 2.
"""

from __future__ import annotations

import numpy as np

from .base import Expert, ExpertAction, MarketState, clip_spread


class GLFTExpert(Expert):
    """Guéant-Lehalle-Fernandez-Tapia market-making strategy.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient (γ > 0).  Default ``0.1``.
    kappa : float
        Base intensity scaling parameter (κ > 0).  Default ``1.5``.
    sigma : float | None
        Volatility override.  If ``None``, uses the volatility from
        :class:`MarketState`.  Default ``None``.
    T : float
        Trading horizon length.  Effective horizon is
        ``T * time_remaining``.  Default ``1.0``.
    intensity_floor : float
        Minimum intensity floor to prevent degenerate spreads when
        arrival rates are near zero (λ > 0 always).  Default ``0.01``.
    clip_lo : float
        Minimum allowed half-spread.  Default ``0.01``.
    clip_hi : float
        Maximum allowed half-spread.  Default ``2.0``.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.5,
        sigma: float | None = None,
        T: float = 1.0,
        intensity_floor: float = 0.01,
        clip_lo: float = 0.01,
        clip_hi: float = 2.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if T <= 0:
            raise ValueError(f"T (horizon) must be positive, got {T}")
        if intensity_floor <= 0:
            raise ValueError(
                f"intensity_floor must be positive, got {intensity_floor}"
            )

        self.gamma: float = gamma
        self.kappa: float = kappa
        self.sigma: float | None = sigma
        self.T: float = T
        self.intensity_floor: float = intensity_floor
        self.clip_lo: float = clip_lo
        self.clip_hi: float = clip_hi

    # ------------------------------------------------------------------
    # Core strategy
    # ------------------------------------------------------------------

    def act(self, state: MarketState) -> ExpertAction:
        """Compute GLFT-optimal bid/ask half-spreads.

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        ExpertAction
            ``delta_bid`` and ``delta_ask`` clipped to ``[clip_lo, clip_hi]``.
        """
        # Resolve volatility
        sigma = self.sigma if self.sigma is not None else state.volatility
        sigma = max(sigma, 1e-8)

        # Effective time to expiry
        tau = self.T * max(state.time_remaining, 0.0)

        # Effective intensities (with floor to avoid zero-division)
        lam_b = max(state.sell_intensity, self.intensity_floor)
        lam_a = max(state.buy_intensity, self.intensity_floor)

        # Intensity parameters κ_b, κ_a
        kappa_b = self.kappa * lam_b
        kappa_a = self.kappa * lam_a

        # Base half-spreads (asymmetric)
        half_spread_b = (1.0 / self.gamma) * np.log(
            1.0 + self.gamma / kappa_b
        )
        half_spread_a = (1.0 / self.gamma) * np.log(
            1.0 + self.gamma / kappa_a
        )

        # Inventory skew: η = q · γ · σ² · τ / 2
        eta = state.inventory * self.gamma * (sigma ** 2) * tau / 2.0

        # Apply inventory adjustment
        delta_bid = half_spread_b + eta
        delta_ask = half_spread_a - eta

        return ExpertAction(
            delta_bid=clip_spread(delta_bid, self.clip_lo, self.clip_hi),
            delta_ask=clip_spread(delta_ask, self.clip_lo, self.clip_hi),
        )

    def name(self) -> str:
        """Return the expert identifier."""
        return "GLFT"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, kappa={self.kappa}, "
            f"sigma={self.sigma}, T={self.T})"
        )


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    expert = GLFTExpert(gamma=0.1, kappa=1.5)

    # --- Test 1: balanced market, neutral inventory ---
    state_balanced = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_balanced)
    print(f"[Balanced]          {expert.name()}: {action}")

    # --- Test 2: buy-heavy market (tight ask, wide bid) ---
    state_buy_heavy = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=3.0,
        sell_intensity=0.5,
        price_change=0.02,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_buy_heavy)
    print(f"[Buy-heavy]         {expert.name()}: {action}")

    # --- Test 3: sell-heavy market (tight bid, wide ask) ---
    state_sell_heavy = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=0.5,
        sell_intensity=3.0,
        price_change=-0.02,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_sell_heavy)
    print(f"[Sell-heavy]        {expert.name()}: {action}")

    # --- Test 4: long inventory with balanced market ---
    state_long = MarketState(
        mid_price=100.0,
        inventory=30,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=0.5,
    )
    action = expert.act(state_long)
    print(f"[Long inventory]    {expert.name()}: {action}")

    # --- Test 5: near-zero intensities (floor kicks in) ---
    state_low_intensity = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=0.0,
        sell_intensity=0.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=1.0,
    )
    action = expert.act(state_low_intensity)
    print(f"[Low intensity]     {expert.name()}: {action}")

    # --- Test 6: near expiry ---
    state_expiry = MarketState(
        mid_price=100.0,
        inventory=10,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=0.0,
    )
    action = expert.act(state_expiry)
    print(f"[Near expiry]       {expert.name()}: {action}")

    print(f"\nRepr: {expert}")
    print("glft.py OK")
