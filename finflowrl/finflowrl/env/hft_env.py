"""Gym-Style HFT Market-Making Environment.

Implements a continuous-action reinforcement-learning environment for
optimal market-making, following the stochastic-control formulation in:

    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)

The agent observes a window of recent market microstructure features and
outputs bid/ask spreads around the mid-price.  The reward combines
realised PnL from limit-order fills with a quadratic inventory-risk
penalty:

    r_t = dCash_t - phi * (I_t^2 - I_{t-1}^2)

At the terminal step the remaining inventory is liquidated at the
mid-price and an additional terminal penalty is applied.

Design Notes
------------
The environment is intentionally self-contained (numpy only) so it can
be used as a drop-in component for any RL framework (Stable-Baselines3,
Ray RLlib, CleanRL, JAX, PyTorch, etc.) by wrapping it in a thin
``gymnasium.Env`` adapter if desired.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .market_simulator import MarketSimulator


class HFTEnv:
    """Gym-style HFT market-making environment.

    The agent observes market state and chooses bid/ask spreads around
    the mid-price.  The goal is to maximise terminal wealth net of
    inventory risk:

        max  E[ W_T - phi * I_T^2  |  O_0 ]

    where ``W_T`` is cumulative cash from fills and ``I_T`` is the
    terminal inventory position.

    Parameters
    ----------
    simulator : MarketSimulator
        An instance of :class:`MarketSimulator` used to drive the
        underlying price and order-flow dynamics.
    max_steps : int
        Maximum number of steps per episode.  Default ``1000``.
    T_obs : int
        Number of past time-steps to include in the observation (lookback
        window).  Default ``2``.
    inventory_limit : int
        Absolute inventory cap; the episode ends early if this is
        breached.  Default ``100``.
    tick_size : float
        Minimum price increment.  Default ``0.01``.
    penalty_phi : float
        Quadratic inventory-risk penalty coefficient.  Default ``0.001``.
    fill_threshold : float
        A market order fills the agent's limit if the agent's price is
        within ``fill_threshold`` ticks of the best bid/ask.  Default
        ``0.5`` (i.e., the agent must be at or better than the mid-point
        between best bid and mid, or best ask and mid).
    """

    # Feature names used in the observation vector (per time-step).
    _FEATURE_NAMES: Tuple[str, ...] = (
        "mid_price",
        "inventory",
        "spread",
        "buy_intensity",
        "sell_intensity",
        "price_change",
        "volatility_estimate",
    )

    def __init__(
        self,
        simulator: MarketSimulator,
        max_steps: int = 1000,
        T_obs: int = 2,
        inventory_limit: int = 100,
        tick_size: float = 0.01,
        penalty_phi: float = 0.001,
        fill_threshold: float = 0.5,
    ) -> None:
        self.simulator = simulator
        self.max_steps = max_steps
        self.T_obs = T_obs
        self.inventory_limit = inventory_limit
        self.tick_size = tick_size
        self.penalty_phi = penalty_phi
        self.fill_threshold = fill_threshold

        # The observation dimension is T_obs * n_features.
        self._n_features = len(self._FEATURE_NAMES)
        self._obs_dim = self.T_obs * self._n_features

        # --- Agent's quoting state ---
        self._current_step: int = 0
        self._inventory: float = 0.0
        self._cash: float = 0.0
        self._prev_mid: float = simulator.S0
        self._prev_inventory: float = 0.0

        # --- Observation history (ring buffer) ---
        self._obs_history: np.ndarray = np.zeros(
            (self.T_obs, self._n_features), dtype=np.float64
        )

        # --- Rolling volatility estimate (exponential moving variance) ---
        self._ewm_variance: float = 0.0
        self._ewm_alpha: float = 0.05  # smoothing factor

        # --- Last market state from simulator ---
        self._last_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Action / Observation space descriptors
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> Dict[str, Any]:
        """Return a dict describing the continuous action space.

        Returns
        -------
        dict
            ``{"shape": (2,), "low": [0.01, 0.01], "high": [2.0, 2.0],
            "dtype": "float32"}``
        """
        return {
            "shape": (2,),
            "low": [0.01, 0.01],
            "high": [2.0, 2.0],
            "dtype": "float32",
        }

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Return a dict describing the observation space.

        Returns
        -------
        dict
            ``{"shape": (obs_dim,), "dtype": "float32"}`` where
            ``obs_dim = T_obs * 7``.
        """
        return {
            "shape": (self._obs_dim,),
            "dtype": "float32",
        }

    @property
    def obs_dim(self) -> int:
        """Observation vector dimension."""
        return self._obs_dim

    # ------------------------------------------------------------------
    # Core Gym-like API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state.

        Returns
        -------
        np.ndarray
            The initial observation vector of shape ``(obs_dim,)``.
        """
        self.simulator.reset()
        self._current_step = 0
        self._inventory = 0.0
        self._cash = 0.0
        self._prev_mid = self.simulator.S0
        self._prev_inventory = 0.0
        self._ewm_variance = 0.0
        self._obs_history = np.zeros(
            (self.T_obs, self._n_features), dtype=np.float64
        )
        self._last_state = None

        # Take one simulator step so that we have a valid market state
        # before the agent acts.
        self._last_state = self.simulator.step()
        self._prev_mid = self._last_state["mid_price"]

        obs = self._get_observation()
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment time-step.

        Parameters
        ----------
        action : np.ndarray
            A 2-element array ``[delta_bid, delta_ask]`` specifying the
            agent's bid/ask half-spreads in price units.  Clipped to
            ``[tick_size, 2.0]``.

        Returns
        -------
        obs : np.ndarray
            Next observation of shape ``(obs_dim,)``.
        reward : float
            Scalar reward for this step.
        done : bool
            Whether the episode has terminated.
        info : dict
            Auxiliary diagnostic information.
        """
        # ---- 0. Store previous inventory for reward calculation ------
        self._prev_inventory = self._inventory

        # ---- 1. Clip & interpret action --------------------------------
        delta_bid = float(np.clip(action[0], self.tick_size, 2.0))
        delta_ask = float(np.clip(action[1], self.tick_size, 2.0))

        if self._last_state is None:
            raise RuntimeError(
                "Environment not initialised.  Call reset() first."
            )

        mid = self._last_state["mid_price"]
        bid_price = mid - delta_bid
        ask_price = mid + delta_ask

        # ---- 2. Advance simulator ------------------------------------
        state = self.simulator.step()
        self._last_state = state

        # ---- 3. Determine fills ---------------------------------------
        # A *market buy* order (aggressor buys) fills the agent's *ask*
        # (agent sells).  Fill occurs probabilistically: the market buy
        # must arrive AND the agent's ask must be competitive (close to
        # or better than the best ask from the simulator).
        #
        # Conversely, a *market sell* order fills the agent's *bid*.

        buy_orders = state["buy_orders"]   # aggressive buy orders arrived
        sell_orders = state["sell_orders"]  # aggressive sell orders arrived

        filled_buy = 0  # agent bought (filled on bid)
        filled_sell = 0  # agent sold (filled on ask)

        sim_ask = state["ask_price"]
        sim_bid = state["bid_price"]

        # Agent sells (ask fill) when aggressive buy arrives and agent's
        # ask is within fill_threshold of the simulator's ask.
        for _ in range(buy_orders):
            # The "distance" between agent's ask and the sim's best ask
            ask_distance = (ask_price - sim_ask) / self.tick_size
            if ask_distance <= self.fill_threshold:
                filled_sell += 1

        # Agent buys (bid fill) when aggressive sell arrives and agent's
        # bid is within fill_threshold of the simulator's bid.
        for _ in range(sell_orders):
            bid_distance = (sim_bid - bid_price) / self.tick_size
            if bid_distance <= self.fill_threshold:
                filled_buy += 1

        # ---- 4. Update inventory & cash -------------------------------
        # Each fill is for 1 unit (could generalise to lot_size).
        self._inventory += filled_buy - filled_sell
        self._cash += filled_sell * ask_price - filled_buy * bid_price

        # ---- 5. Calculate reward --------------------------------------
        reward = self._calculate_reward(filled_buy, filled_sell, bid_price, ask_price)

        # ---- 6. Update rolling variance estimate ----------------------
        price_change = state["mid_price"] - self._prev_mid
        self._prev_mid = state["mid_price"]
        self._ewm_variance = (
            self._ewm_alpha * price_change ** 2
            + (1.0 - self._ewm_alpha) * self._ewm_variance
        )

        # ---- 7. Build observation -------------------------------------
        obs = self._get_observation()

        # ---- 8. Check termination -------------------------------------
        done = False
        inventory_breach = abs(self._inventory) > self.inventory_limit
        step_limit = self._current_step >= self.max_steps
        if inventory_breach or step_limit:
            done = True
            # Terminal liquidation: close remaining inventory at mid-price
            terminal_pnl = -self._inventory * state["mid_price"]
            terminal_penalty = (
                -self.penalty_phi * self._inventory ** 2
            )
            reward += terminal_pnl + terminal_penalty

        self._current_step += 1

        info = {
            "step": self._current_step,
            "inventory": self._inventory,
            "cash": self._cash,
            "mid_price": state["mid_price"],
            "filled_buy": filled_buy,
            "filled_sell": filled_sell,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "inventory_breach": inventory_breach,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the flattened observation vector.

        The observation consists of ``T_obs`` stacked feature vectors
        (oldest first).  If fewer than ``T_obs`` steps have elapsed the
        remaining rows are zero-padded (as set in ``reset``).

        Features per time-step (normalised where appropriate):
            0. mid_price       – raw (the agent learns the scale)
            1. inventory       – raw (signed)
            2. spread          – raw
            3. buy_intensity   – raw
            4. sell_intensity  – raw
            5. price_change    – mid_price_t - mid_price_{t-1}
            6. volatility_estimate – sqrt(EWM variance)

        Returns
        -------
        np.ndarray of shape ``(obs_dim,)``
        """
        if self._last_state is None:
            return np.zeros(self._obs_dim, dtype=np.float32)

        s = self._last_state
        price_change = s["mid_price"] - self._prev_mid
        vol_estimate = math.sqrt(max(self._ewm_variance, 0.0))

        features = np.array(
            [
                s["mid_price"],
                self._inventory,
                s["spread"],
                s["buy_intensity"],
                s["sell_intensity"],
                price_change,
                vol_estimate,
            ],
            dtype=np.float64,
        )

        # Shift history and append newest row
        self._obs_history = np.roll(self._obs_history, -1, axis=0)
        self._obs_history[-1] = features

        # Flatten: (T_obs, n_features) -> (T_obs * n_features,)
        return self._obs_history.flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        filled_buy: int,
        filled_sell: int,
        bid_price: float,
        ask_price: float,
    ) -> float:
        """Compute the one-step reward.

        The reward has two components:

        1. **Mark-to-market PnL change**: change in cash from fills plus
           the change in mark-to-market value of inventory.

        2. **Inventory risk penalty**: quadratic penalty on inventory
           deviation, penalising the *change* in penalty rather than the
           absolute level (to avoid double-counting):

               -phi * (I_t^2 - I_{t-1}^2)

        Parameters
        ----------
        filled_buy : int
            Number of units the agent bought (bid fills).
        filled_sell : int
            Number of units the agent sold (ask fills).
        bid_price : float
            Agent's bid price this step.
        ask_price : float
            Agent's ask price this step.

        Returns
        -------
        float
            The scalar reward.
        """
        # Cash PnL from fills: sell at ask, buy at bid
        cash_pnl = filled_sell * ask_price - filled_buy * bid_price

        # Mark-to-market change in inventory value
        if self._last_state is not None:
            mid = self._last_state["mid_price"]
        else:
            mid = self.simulator.S0

        mtm_change = self._inventory * (
            mid - self._prev_mid
        ) if self._prev_mid != 0.0 else 0.0

        # Inventory risk penalty (penalise change to avoid accumulation)
        inventory_penalty = -self.penalty_phi * (
            self._inventory ** 2 - self._prev_inventory ** 2
        )

        reward = cash_pnl + mtm_change + inventory_penalty
        return float(reward)
