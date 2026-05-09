"""GLFT with price-drift extension for market-making.

Extends the Guéant-Lehalle-Fernandez-Tapia (GLFT) strategy with an
explicit price-drift term (μ).  When the mid-price exhibits a directional
trend, the drift adjustment shifts the quoting centre, causing the
market-maker to lean into the trend:

    * Positive drift (up-trending) → widen bid, tighten ask (sell into
      strength, reluctant to buy).
    * Negative drift (down-trending) → tighten bid, widen ask (buy into
      weakness, reluctant to sell).

The drift estimate is derived from the ``price_change`` and ``buy_intensity``
/ ``sell_intensity`` imbalance observed in the :class:`MarketState`.

Key formulas
------------
Start with the standard GLFT half-spreads (δ_b, δ_a) then add:

    drift_adj = μ · (T − t)

    δ_b ← δ_b + drift_adj
    δ_a ← δ_a − drift_adj

The drift μ is estimated as a blend of:
    1. *Signed price change*: recent mid-price momentum.
    2. *Intensity imbalance*: ``(λ⁺ − λ⁻) / (λ⁺ + λ⁻)`` – captures
       order-flow pressure that may not yet be reflected in price.

Reference
---------
    FinFlowRL (arXiv 2509.17964) – Section 3.3, Algorithm 3.
"""

from __future__ import annotations

import numpy as np

from .base import Expert, ExpertAction, MarketState, clip_spread
from .glft import GLFTExpert


class GLFTDriftExpert(GLFTExpert):
    """GLFT market-making strategy with price-drift adjustment.

    Inherits all parameters from :class:`GLFTExpert` and adds:

    Parameters
    ----------
    mu : float | None
        Explicit drift coefficient.  If ``None`` (default), the expert
        auto-estimates drift from ``price_change`` and the intensity
        imbalance at each step.
    drift_blend : float
        Weight for blending the auto-estimated drift with the explicit μ.
        Only used when ``mu`` is not ``None``.  The effective drift is:

            μ_eff = drift_blend · μ + (1 − drift_blend) · μ_estimated

        Ignored when ``mu is None``.  Default ``0.5``.
    drift_cap : float
        Maximum absolute value of the drift adjustment to prevent
        extreme quoting.  Default ``0.5``.
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
        mu: float | None = None,
        drift_blend: float = 0.5,
        drift_cap: float = 0.5,
    ) -> None:
        super().__init__(
            gamma=gamma,
            kappa=kappa,
            sigma=sigma,
            T=T,
            intensity_floor=intensity_floor,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )

        if mu is not None and not np.isfinite(mu):
            raise ValueError(f"mu must be finite, got {mu}")
        if not (0.0 <= drift_blend <= 1.0):
            raise ValueError(
                f"drift_blend must be in [0, 1], got {drift_blend}"
            )
        if drift_cap < 0:
            raise ValueError(f"drift_cap must be non-negative, got {drift_cap}")

        self.mu: float | None = mu
        self.drift_blend: float = drift_blend
        self.drift_cap: float = drift_cap

    # ------------------------------------------------------------------
    # Drift estimation
    # ------------------------------------------------------------------

    def _estimate_drift(self, state: MarketState) -> float:
        """Auto-estimate the price drift from market observables.

        Combines two signals:

        1. **Price momentum**: ``price_change`` directly.
        2. **Order-flow imbalance**:
           ``(buy_intensity − sell_intensity) / (buy_intensity + sell_intensity)``
           scaled by ``volatility`` to put it in comparable units.

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        float
            Estimated drift μ.
        """
        # Signal 1: raw price change
        momentum = state.price_change

        # Signal 2: intensity imbalance
        total_intensity = state.buy_intensity + state.sell_intensity
        if total_intensity > self.intensity_floor:
            imbalance = (
                (state.buy_intensity - state.sell_intensity)
                / total_intensity
            )
        else:
            imbalance = 0.0

        # Scale imbalance by volatility so the units are comparable
        vol = max(state.volatility, 1e-8)
        flow_signal = imbalance * vol

        # Blend: equal weight by default
        mu_est = 0.5 * momentum + 0.5 * flow_signal

        return float(mu_est)

    # ------------------------------------------------------------------
    # Core strategy
    # ------------------------------------------------------------------

    def act(self, state: MarketState) -> ExpertAction:
        """Compute GLFT+drift optimal bid/ask half-spreads.

        Steps:
        1. Get the base GLFT half-spreads.
        2. Estimate (or use explicit) drift μ.
        3. Apply ``drift_adj = μ · (T − t)`` symmetrically.

        Parameters
        ----------
        state : MarketState
            Current market snapshot.

        Returns
        -------
        ExpertAction
            ``delta_bid`` and ``delta_ask`` clipped to ``[clip_lo, clip_hi]``.
        """
        # Step 1: base GLFT spreads
        base_action = super().act(state)

        # Step 2: resolve drift
        mu_est = self._estimate_drift(state)

        if self.mu is not None:
            mu_eff = (
                self.drift_blend * self.mu
                + (1.0 - self.drift_blend) * mu_est
            )
        else:
            mu_eff = mu_est

        # Cap drift to prevent extreme adjustments
        mu_eff = float(np.clip(mu_eff, -self.drift_cap, self.drift_cap))

        # Step 3: drift adjustment scaled by remaining time
        tau = self.T * max(state.time_remaining, 0.0)
        drift_adj = mu_eff * tau

        # Positive drift → widen bid, tighten ask (lean into uptrend)
        delta_bid = base_action.delta_bid + drift_adj
        delta_ask = base_action.delta_ask - drift_adj

        return ExpertAction(
            delta_bid=clip_spread(delta_bid, self.clip_lo, self.clip_hi),
            delta_ask=clip_spread(delta_ask, self.clip_lo, self.clip_hi),
        )

    def name(self) -> str:
        """Return the expert identifier."""
        return "GLFT-Drift"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, kappa={self.kappa}, "
            f"sigma={self.sigma}, T={self.T}, "
            f"mu={self.mu}, drift_cap={self.drift_cap})"
        )


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    expert = GLFTDriftExpert(gamma=0.1, kappa=1.5)

    # --- Test 1: neutral – no price change, balanced intensity ---
    state_neutral = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=1.0,
        sell_intensity=1.0,
        price_change=0.0,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_neutral)
    print(f"[Neutral]           {expert.name()}: {action}")

    # --- Test 2: uptrend (positive drift) ---
    state_up = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=2.0,
        sell_intensity=0.5,
        price_change=0.05,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_up)
    print(f"[Uptrend]           {expert.name()}: {action}")

    # --- Test 3: downtrend (negative drift) ---
    state_down = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=0.5,
        sell_intensity=2.0,
        price_change=-0.05,
        volatility=0.02,
        time_remaining=0.75,
    )
    action = expert.act(state_down)
    print(f"[Downtrend]         {expert.name()}: {action}")

    # --- Test 4: explicit mu ---
    expert_explicit = GLFTDriftExpert(gamma=0.1, kappa=1.5, mu=0.02)
    action_explicit = expert_explicit.act(state_neutral)
    print(f"[Explicit μ=0.02]   {expert_explicit.name()}: {action_explicit}")

    # --- Test 5: large drift (should be capped) ---
    state_extreme = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=10.0,
        sell_intensity=0.0,
        price_change=1.0,
        volatility=0.02,
        time_remaining=1.0,
    )
    action_extreme = expert.act(state_extreme)
    print(f"[Extreme drift]     {expert.name()}: {action_extreme}")

    # --- Test 6: near expiry (drift adj → 0) ---
    state_expiry = MarketState(
        mid_price=100.0,
        inventory=0,
        spread=0.05,
        buy_intensity=2.0,
        sell_intensity=0.5,
        price_change=0.05,
        volatility=0.02,
        time_remaining=0.0,
    )
    action_expiry = expert.act(state_expiry)
    print(f"[Near expiry]       {expert.name()}: {action_expiry}")

    print(f"\nRepr: {expert}")
    print("glft_drift.py OK")
