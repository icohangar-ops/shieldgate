"""Market Microstructure Simulator for HFT Research.

Implements a realistic market simulator based on the formulation in:

    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)

The simulator combines:
    1. **Jump-Diffusion Price Process** (Merton 1976) with fractional
       Brownian motion for realistic price dynamics including long-range
       dependence and sudden jumps.
    2. **Bivariate Hawkes Process** for self/cross-exciting order arrivals
       capturing the clustering of market orders and cross-side excitation
       typical in LOB microstructure.

References
----------
    - Merton, R. C. (1976). "Option pricing when underlying stock returns
      are discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
    - Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually
      exciting point processes." *Biometrika*, 58(1), 83-90.
    - Mandelbrot, B. B. & Van Ness, J. W. (1968). "Fractional Brownian
      motions, fractional noises and applications." *SIAM Review*, 10(4).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MarketSimulator:
    """Simulate realistic market microstructure for HFT research.

    The price process follows a jump-diffusion model with fractional
    Brownian motion (fBm):

        dS_t = S_{t-} [mu * dt + sigma * dB_H(t)] + S_{t-} * (e^J - 1) * dN_t

    where ``J ~ N(mu_J, sigma_J^2)`` and ``dN_t`` is a Poisson process
    with intensity ``lambda_J``.

    Order arrivals are modelled as a bivariate Hawkes process:

        lambda_a(t) = mu_a
                    + sum_{t_i in N_a} alpha_aa * exp(-beta * (t - t_i))
                    + sum_{t_j in N_b} alpha_ab * exp(-beta * (t - t_j))

        lambda_b(t) = mu_b
                    + sum_{t_i in N_b} alpha_bb * exp(-beta * (t - t_i))
                    + sum_{t_j in N_a} alpha_ba * exp(-beta * (t - t_j))

    Parameters
    ----------
    S0 : float
        Initial mid-price.  Default ``100.0``.
    mu : float
        Drift rate of the price process.  Default ``0.0``.
    sigma : float
        Volatility of the price diffusion.  Default ``0.1``.
    H : float
        Hurst exponent for fractional Brownian motion.
        ``H = 0.5`` gives standard BM; ``H > 0.5`` gives persistent
        (trend-following) increments; ``H < 0.5`` gives mean-reverting
        increments.  Default ``0.5``.
    mu_J : float
        Mean of the jump-size distribution.  Default ``-0.02``.
    sigma_J : float
        Std-dev of the jump-size distribution.  Default ``0.03``.
    lambda_J : float
        Jump arrival intensity (Poisson rate).  Default ``0.1``.
    mu_a : float
        Baseline arrival rate for *aggressive buy* (market-buy) orders.
        Default ``10.0``.
    mu_b : float
        Baseline arrival rate for *aggressive sell* (market-sell) orders.
        Default ``10.0``.
    alpha_aa : float
        Self-excitation magnitude for buy orders.  Default ``5.0``.
    alpha_ab : float
        Cross-excitation: impact of sell-order arrivals on buy intensity.
        Default ``3.0``.
    alpha_bb : float
        Self-excitation magnitude for sell orders.  Default ``5.0``.
    alpha_ba : float
        Cross-excitation: impact of buy-order arrivals on sell intensity.
        Default ``3.0``.
    beta : float
        Exponential decay rate for Hawkes excitation.  Default ``10.0``.
    dt : float
        Time-step size.  Default ``0.01``.
    seed : int or None
        Random seed for reproducibility.  Default ``None``.
    """

    def __init__(
        self,
        S0: float = 100.0,
        mu: float = 0.0,
        sigma: float = 0.1,
        H: float = 0.5,
        mu_J: float = -0.02,
        sigma_J: float = 0.03,
        lambda_J: float = 0.1,
        mu_a: float = 10.0,
        mu_b: float = 10.0,
        alpha_aa: float = 5.0,
        alpha_ab: float = 3.0,
        alpha_bb: float = 5.0,
        alpha_ba: float = 3.0,
        beta: float = 10.0,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        # --- Price process parameters ---
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.H = H
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.lambda_J = lambda_J

        # --- Hawkes process parameters ---
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.alpha_aa = alpha_aa
        self.alpha_ab = alpha_ab
        self.alpha_bb = alpha_bb
        self.alpha_ba = alpha_ba
        self.beta = beta

        # --- Simulation parameters ---
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # --- Internal state ---
        self._current_time: float = 0.0
        self._mid_price: float = S0
        self._buy_intensity: float = mu_a
        self._sell_intensity: float = mu_b

        # History of order arrival times for Hawkes kernel evaluation.
        # Each is a list of relative timestamps (seconds since start).
        self._buy_order_times: List[float] = []
        self._sell_order_times: List[float] = []

        # Pre-compute the fBm covariance coefficients for the Cholesky /
        # circulant-embedding approach (lazily on first call to ``step``).
        self._fbm_buffer: Optional[np.ndarray] = None
        self._fbm_idx: int = 0
        self._fbm_buffer_size: int = 512  # regenerate in blocks

        # Semi-variance cache for the fBm approximation (Hosking-style).
        self._gamma_cache: Optional[np.ndarray] = None

        # Spread model: half-spread as a fraction of mid-price.
        self._base_half_spread: float = 0.0005  # 5 bps

        # Reset to initial state
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset the simulator to its initial state.

        Returns
        -------
        dict
            The initial state dictionary (same format as :meth:`step`).
        """
        self._current_time = 0.0
        self._mid_price = self.S0
        self._buy_intensity = self.mu_a
        self._sell_intensity = self.mu_b
        self._buy_order_times: List[float] = []
        self._sell_order_times: List[float] = []
        self._fbm_buffer = None
        self._fbm_idx = 0
        self._gamma_cache = None

        return self._make_state(
            jump_occurred=False,
            buy_orders=0,
            sell_orders=0,
            inventory_delta=0,
        )

    def step(self) -> Dict[str, Any]:
        """Advance the simulator by one time-step ``dt``.

        Returns
        -------
        dict
            A dictionary containing:
            - ``mid_price``  : float  – current mid-price
            - ``bid_price``  : float  – best bid
            - ``ask_price``  : float  – best ask
            - ``spread``     : float  – bid-ask spread
            - ``buy_intensity`` : float – current buy-order arrival rate
            - ``sell_intensity``: float – current sell-order arrival rate
            - ``buy_orders`` : int    – number of buy orders this step
            - ``sell_orders`` : int   – number of sell orders this step
            - ``jump_occurred`` : bool – whether a price jump happened
            - ``inventory_delta`` : float – net change in implied inventory
        """
        dt = self.dt
        t = self._current_time

        # ---- 1. Generate fBm increment --------------------------------
        fbm_inc = self._next_fbm_increment()

        # ---- 2. Jump component ----------------------------------------
        # Poisson jump: number of jumps in [t, t+dt)
        n_jumps = self.rng.poisson(self.lambda_J * dt)
        jump_occurred = n_jumps > 0
        jump_factor = 0.0
        if jump_occurred:
            # Compound Poisson: sum of exp(J_i) - 1 for each jump
            jump_sizes = self.rng.normal(self.mu_J, self.sigma_J, size=n_jumps)
            jump_factor = float(np.sum(np.exp(jump_sizes) - 1.0))

        # ---- 3. Update mid-price (Euler-Maruyama) ---------------------
        dS = (
            self._mid_price * self.mu * dt
            + self._mid_price * self.sigma * math.sqrt(dt) * fbm_inc
            + self._mid_price * jump_factor
        )
        self._mid_price = max(self._mid_price + dS, 1e-8)  # floor at epsilon

        # ---- 4. Hawkes intensities ------------------------------------
        self._buy_intensity = self._compute_hawkes_intensity("buy", t)
        self._sell_intensity = self._compute_hawkes_intensity("sell", t)

        # ---- 5. Simulate order arrivals (thinning approximation) ------
        buy_orders = self.rng.poisson(max(self._buy_intensity * dt, 0.0))
        sell_orders = self.rng.poisson(max(self._sell_intensity * dt, 0.0))

        # Record arrival times for future kernel evaluation
        if buy_orders > 0:
            # Spread arrivals uniformly within [t, t+dt)
            self._buy_order_times.extend(
                (t + self.rng.uniform(0, dt, size=buy_orders)).tolist()
            )
        if sell_orders > 0:
            self._sell_order_times.extend(
                (t + self.rng.uniform(0, dt, size=sell_orders)).tolist()
            )

        # Prune very old arrival times to keep memory bounded
        cutoff = t - 5.0 / self.beta  # ~5 half-lives
        self._buy_order_times = [ti for ti in self._buy_order_times if ti > cutoff]
        self._sell_order_times = [ti for ti in self._sell_order_times if ti > cutoff]

        # ---- 6. Inventory delta (for downstream use) ------------------
        inventory_delta = int(buy_orders) - int(sell_orders)

        # ---- 7. Advance clock -----------------------------------------
        self._current_time = t + dt

        return self._make_state(
            jump_occurred=jump_occurred,
            buy_orders=int(buy_orders),
            sell_orders=int(sell_orders),
            inventory_delta=inventory_delta,
        )

    def generate_trajectory(self, n_steps: int) -> List[Dict[str, Any]]:
        """Generate a full trajectory of ``n_steps`` observations.

        Parameters
        ----------
        n_steps : int
            Number of time-steps to simulate.

        Returns
        -------
        list of dict
            A list of state dictionaries, one per time-step.
        """
        self.reset()
        trajectory: List[Dict[str, Any]] = []
        for _ in range(n_steps):
            state = self.step()
            trajectory.append(state)
        return trajectory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_state(
        self,
        jump_occurred: bool,
        buy_orders: int,
        sell_orders: int,
        inventory_delta: int,
    ) -> Dict[str, Any]:
        """Construct the output state dictionary for the current tick."""
        # Stochastic half-spread: add noise proportional to recent intensity.
        total_intensity = max(self._buy_intensity + self._sell_intensity, 1e-8)
        noise = self.rng.normal(0.0, 0.1) * self._base_half_spread * math.sqrt(
            10.0 / total_intensity
        )
        half_spread = max(self._base_half_spread + noise, 1e-6)

        bid_price = self._mid_price - half_spread
        ask_price = self._mid_price + half_spread
        spread = ask_price - bid_price

        return {
            "mid_price": float(self._mid_price),
            "bid_price": float(bid_price),
            "ask_price": float(ask_price),
            "spread": float(spread),
            "buy_intensity": float(self._buy_intensity),
            "sell_intensity": float(self._sell_intensity),
            "buy_orders": buy_orders,
            "sell_orders": sell_orders,
            "jump_occurred": jump_occurred,
            "inventory_delta": inventory_delta,
        }

    # ---- Fractional Brownian Motion -----------------------------------

    def _build_fbm_covariance(self, n: int) -> np.ndarray:
        """Build the covariance matrix for n fBm increments.

        Uses the autocovariance of fBm *increment* process:
            gamma(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})

        For ``H = 0.5`` this gives delta(k) (standard BM).

        Parameters
        ----------
        n : int
            Size of the covariance matrix.

        Returns
        -------
        np.ndarray of shape (n, n)
            Symmetric positive-(semi)definite covariance matrix.
        """
        H = self.H
        indices = np.arange(n)
        # Vectorised computation of gamma(|i-j|)
        diff = np.abs(indices[:, None] - indices[None, :])
        gamma = 0.5 * (
            np.abs(diff - 1) ** (2 * H)
            - 2.0 * np.abs(diff) ** (2 * H)
            + np.abs(diff + 1) ** (2 * H)
        )
        return gamma

    def _ensure_fbm_buffer(self) -> None:
        """Lazily allocate / refill the fBm increment buffer."""
        if self._fbm_buffer is not None and self._fbm_idx < self._fbm_buffer_size:
            return

        size = self._fbm_buffer_size
        cov = self._build_fbm_covariance(size)

        # Add small jitter on the diagonal for numerical stability
        cov += np.eye(size) * 1e-10

        try:
            L = np.linalg.cholesky(cov)
            z = self.rng.standard_normal(size)
            self._fbm_buffer = L @ z
        except np.linalg.LinAlgError:
            # Fallback: use eigen-decomposition for near-singular cases
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            z = self.rng.standard_normal(size)
            self._fbm_buffer = L @ z

        self._fbm_idx = 0

    def _next_fbm_increment(self) -> float:
        """Return the next pre-generated fBm increment."""
        self._ensure_fbm_buffer()
        val = float(self._fbm_buffer[self._fbm_idx])
        self._fbm_idx += 1
        return val

    # ---- Hawkes Process -----------------------------------------------

    def _compute_hawkes_intensity(
        self, side: str, current_time: float
    ) -> float:
        """Evaluate the Hawkes conditional intensity at *current_time*.

        Parameters
        ----------
        side : str
            ``"buy"`` or ``"sell"``.
        current_time : float
            Current simulation time.

        Returns
        -------
        float
            The conditional intensity lambda(t) for the given side.
        """
        if side == "buy":
            mu = self.mu_a
            own_times = self._buy_order_times
            cross_times = self._sell_order_times
            alpha_own = self.alpha_aa
            alpha_cross = self.alpha_ab
        else:
            mu = self.mu_b
            own_times = self._sell_order_times
            cross_times = self._buy_order_times
            alpha_own = self.alpha_bb
            alpha_cross = self.alpha_ba

        intensity = mu

        # Self-excitation kernel
        for ti in own_times:
            intensity += alpha_own * math.exp(-self.beta * (current_time - ti))

        # Cross-excitation kernel
        for ti in cross_times:
            intensity += alpha_cross * math.exp(-self.beta * (current_time - ti))

        return max(intensity, 0.0)  # intensity must be non-negative
