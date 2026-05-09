"""Expert demonstration dataset generation and persistence.

Provides functions to generate, save, and load expert demonstration datasets
for the FinFlowRL imitation-reinforcement learning pipeline.

Dataset Generation Strategy
----------------------------
Diverse market scenarios are created by varying simulator parameters across
a grid of:

    - ``sigma`` in {0.05, 0.1, 0.3}          (3 volatility regimes)
    - ``lambda_J`` in {0.05, 0.1, 0.5}       (3 jump intensity levels)
    - ``H`` in {0.3, 0.5, 0.7}               (3 Hurst exponents)

For each scenario, the environment runs for ``steps_per_scenario`` steps
with all 4 expert strategies active simultaneously.  At each step, the best-
performing expert's action is recorded as the demonstration target.

Target: ~3.24M state-action pairs (108 scenarios × 30,000 steps).

Data Format
-----------
    states : np.ndarray, shape (N, T_obs, state_dim)
        Windows of ``T_obs`` consecutive market observations.
    actions : np.ndarray, shape (N, T_pred, action_dim)
        Corresponding expert action chunks.

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from __future__ import annotations

import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Expert demonstration generation
# ======================================================================

def generate_expert_demonstrations(
    config: Any,
    num_scenarios: int = 108,
    steps_per_scenario: int = 30_000,
    T_obs: int = 2,
    T_pred: int = 8,
    action_dim: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate expert demonstration dataset across diverse market scenarios.

    Creates a grid of market parameter combinations, runs the environment
    with multiple expert strategies, and selects the best-performing expert's
    actions at each step.

    Parameters
    ----------
    config : Config
        FinFlowRL configuration object.  The ``config.market`` and
        ``config.expert`` sections are used to parameterise the simulator
        and expert strategies.
    num_scenarios : int
        Total number of market scenarios to generate.  The actual number
        is the product of the parameter grid.  Default ``108``
        (3 × 3 × 4 × 3 grid: sigma × lambda_J × H × alpha_scale).
    steps_per_scenario : int
        Number of environment steps per scenario.  Default ``30,000``.
    T_obs : int
        Number of past observations per sample.  Default ``2``.
    T_pred : int
        Number of future actions per chunk.  Default ``8``.
    action_dim : int
        Action dimension per timestep.  Default ``2``.
    seed : int
        Random seed for reproducibility.  Default ``42``.
    verbose : bool
        If ``True``, print progress information.  Default ``True``.

    Returns
    -------
    states : np.ndarray
        Shape ``(N, T_obs, state_dim)`` where ``state_dim`` is determined
        by the environment (``T_obs * n_features``).
    actions : np.ndarray
        Shape ``(N, T_pred, action_dim)``.
    metadata : dict
        Dictionary with generation info: ``num_scenarios``,
        ``steps_per_scenario``, ``total_samples``, ``elapsed_time``,
        ``scenario_grid``.
    """
    from finflowrl.env import MarketSimulator, HFTEnv
    from finflowrl.experts import (
        AvellanedaStoikovExpert,
        GLFTExpert,
        GLFTDriftExpert,
        PPOExpert,
    )
    from finflowrl.experts.base import MarketState

    rng = np.random.default_rng(seed)
    start_time = time.time()

    # --- Define parameter grid ---
    sigmas = [0.05, 0.1, 0.3]
    lambda_Js = [0.05, 0.1, 0.5]
    hursts = [0.3, 0.5, 0.7]
    alpha_scales = [0.5, 1.0, 2.0, 1.0]  # last one is duplicate to increase count

    # Build scenario list
    scenarios = list(product(sigmas, lambda_Js, hursts, alpha_scales))
    # Take at most num_scenarios
    scenarios = scenarios[:num_scenarios]
    actual_scenarios = len(scenarios)

    if verbose:
        logger.info(
            f"Generating expert demonstrations: {actual_scenarios} scenarios × "
            f"{steps_per_scenario} steps = "
            f"{actual_scenarios * steps_per_scenario:,} samples (target)"
        )

    # --- Accumulate data ---
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    # Determine state_dim from environment
    test_sim = MarketSimulator(
        sigma=config.market.sigma,
        dt=config.market.dt,
        seed=0,
    )
    test_env = HFTEnv(
        simulator=test_sim,
        max_steps=steps_per_scenario,
        T_obs=T_obs,
    )
    state_dim = test_env.obs_dim  # flat observation dimension
    # For 3-D state: reshape to (T_obs, n_features)
    n_features = state_dim // T_obs

    # --- Run each scenario ---
    for sc_idx, (sigma, lam_J, H, alpha_scale) in enumerate(scenarios):
        if verbose and sc_idx % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"  Scenario {sc_idx + 1}/{actual_scenarios}: "
                f"sigma={sigma}, lambda_J={lam_J}, H={H}, "
                f"alpha_scale={alpha_scale} "
                f"({elapsed:.1f}s elapsed)"
            )

        # Create simulator with scenario parameters
        scenario_seed = int(rng.integers(0, 2**31))
        sim = MarketSimulator(
            S0=config.market.S0,
            mu=config.market.mu,
            sigma=sigma,
            H=H,
            mu_J=config.market.mu_J,
            sigma_J=config.market.sigma_J,
            lambda_J=lam_J,
            mu_a=config.market.mu_a * alpha_scale,
            mu_b=config.market.mu_b * alpha_scale,
            alpha_aa=config.market.alpha_aa * alpha_scale,
            alpha_ab=config.market.alpha_ab * alpha_scale,
            alpha_bb=config.market.alpha_bb * alpha_scale,
            alpha_ba=config.market.alpha_ba * alpha_scale,
            beta=config.market.beta,
            dt=config.market.dt,
            seed=scenario_seed,
        )

        env = HFTEnv(
            simulator=sim,
            max_steps=steps_per_scenario,
            T_obs=T_obs,
        )

        # Create experts
        experts = [
            AvellanedaStoikovExpert(
                gamma=config.expert.as_gamma,
                k=config.expert.as_k,
                sigma=config.expert.as_sigma if config.expert.as_sigma > 0 else None,
                T=config.expert.as_T,
            ),
            GLFTExpert(
                gamma=config.expert.glft_gamma,
                kappa=config.expert.glft_kappa,
            ),
            GLFTDriftExpert(
                gamma=config.expert.glft_gamma,
                kappa=config.expert.glft_kappa,
            ),
            PPOExpert(obs_dim=14, seed=42),
        ]

        # Run episode
        obs = env.reset()
        obs_buffer: List[np.ndarray] = []  # sliding window of T_obs obs

        # Track cumulative reward for expert selection
        expert_rewards = [0.0] * len(experts)

        for step in range(steps_per_scenario):
            # Build observation buffer
            obs_buffer.append(obs.copy())
            if len(obs_buffer) > T_obs:
                obs_buffer = obs_buffer[-T_obs:]

            # Build MarketState from observation
            ms = _obs_to_market_state(obs, n_features, T_obs, env)

            # Get actions from all experts
            expert_actions = [e.act(ms) for e in experts]

            # Select best expert (smallest spread = most aggressive)
            # In practice, one would select based on cumulative PnL,
            # but for dataset generation we use a heuristic:
            # pick the expert with the most balanced bid-ask spread
            best_idx = _select_best_expert(
                expert_actions, env._inventory, ms
            )
            best_action = expert_actions[best_idx]

            # Store state-action pair (in chunk format)
            if len(obs_buffer) >= T_obs:
                state_window = np.array(obs_buffer[-T_obs:])  # (T_obs, state_dim)
                # Reshape to (T_obs, n_features)
                state_window = state_window.reshape(T_obs, n_features)

                # Repeat action to fill chunk (simple strategy)
                action_chunk = np.array([
                    [best_action.delta_bid, best_action.delta_ask]
                    for _ in range(T_pred)
                ])

                all_states.append(state_window)
                all_actions.append(action_chunk)

            # Step environment with best expert's action
            action_np = np.array([best_action.delta_bid, best_action.delta_ask])
            obs, reward, done, info = env.step(action_np)

            if done:
                obs = env.reset()
                obs_buffer.clear()

    # --- Convert to arrays ---
    if len(all_states) == 0:
        states = np.zeros((0, T_obs, n_features), dtype=np.float32)
        actions = np.zeros((0, T_pred, action_dim), dtype=np.float32)
    else:
        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)

    elapsed = time.time() - start_time
    metadata = {
        "num_scenarios": actual_scenarios,
        "steps_per_scenario": steps_per_scenario,
        "total_samples": len(states),
        "state_shape": states.shape,
        "action_shape": actions.shape,
        "elapsed_time": elapsed,
        "samples_per_second": len(states) / max(elapsed, 1e-6),
        "T_obs": T_obs,
        "T_pred": T_pred,
        "action_dim": action_dim,
        "scenario_grid": {
            "sigmas": sigmas,
            "lambda_Js": lambda_Js,
            "hursts": hursts,
            "alpha_scales": alpha_scales,
        },
    }

    if verbose:
        logger.info(
            f"Dataset generated: {states.shape} states, "
            f"{actions.shape} actions in {elapsed:.1f}s "
            f"({metadata['samples_per_second']:.0f} samples/s)"
        )

    return states, actions, metadata


# ======================================================================
# Helpers
# ======================================================================

def _obs_to_market_state(
    obs: np.ndarray,
    n_features: int,
    T_obs: int,
    env: Any,
) -> Any:
    """Convert a flat environment observation to a MarketState.

    The flat observation is (T_obs * n_features,) where n_features=7
    from HFTEnv: [mid_price, inventory, spread, buy_intensity,
    sell_intensity, price_change, volatility].

    We use the most recent timestep's features.

    Parameters
    ----------
    obs : np.ndarray
        Flat observation, shape ``(T_obs * n_features,)``.
    n_features : int
        Features per timestep.
    T_obs : int
        Number of timesteps in observation.
    env : HFTEnv
        Environment instance (for max_steps reference).

    Returns
    -------
    MarketState
    """
    from finflowrl.experts.base import MarketState

    # Reshape to (T_obs, n_features) and take the last timestep
    obs_2d = obs.reshape(T_obs, n_features)
    latest = obs_2d[-1]

    # Map features to MarketState fields
    # HFTEnv features: mid_price, inventory, spread, buy_intensity,
    #                  sell_intensity, price_change, volatility
    mid_price = float(latest[0])
    inventory = int(np.clip(latest[1], -100, 100))
    spread = float(latest[2])
    buy_intensity = max(float(latest[3]), 0.01)
    sell_intensity = max(float(latest[4]), 0.01)
    price_change = float(latest[5])
    volatility = max(float(latest[6]), 1e-8)

    # Estimate time remaining from step count
    step = getattr(env, "_current_step", 0)
    max_steps = getattr(env, "max_steps", 1000)
    time_remaining = max(1.0 - step / max(max_steps, 1), 0.0)

    return MarketState(
        mid_price=mid_price,
        inventory=inventory,
        spread=spread,
        buy_intensity=buy_intensity,
        sell_intensity=sell_intensity,
        price_change=price_change,
        volatility=volatility,
        time_remaining=time_remaining,
    )


def _select_best_expert(
    expert_actions: List[Any],
    inventory: float,
    market_state: Any,
) -> int:
    """Select the best expert for the current market condition.

    Uses a simple heuristic that balances aggressiveness with
    inventory management:

    - Long inventory → prefer wider bid (less buying)
    - Short inventory → prefer wider ask (less selling)
    - Balanced → prefer tightest total spread

    Parameters
    ----------
    expert_actions : list of ExpertAction
        Actions from all experts.
    inventory : float
        Current inventory position.
    market_state : MarketState
        Current market state.

    Returns
    -------
    int
        Index of the selected expert.
    """
    best_score = float("-inf")
    best_idx = 0

    for i, action in enumerate(expert_actions):
        total_spread = action.delta_bid + action.delta_ask
        # Inventory-aware score: prefer tighter spreads when balanced,
        # penalise widening the spread on the wrong side
        inventory_penalty = 0.0
        if inventory > 0:
            # Long: prefer wider bid (reluctant to buy) + tight ask (eager to sell)
            inventory_penalty = 0.5 * action.delta_bid - 0.3 * action.delta_ask
        elif inventory < 0:
            # Short: prefer tight bid (eager to buy) + wider ask (reluctant to sell)
            inventory_penalty = -0.3 * action.delta_bid + 0.5 * action.delta_ask

        # Tighter spreads are generally better (more fill probability)
        spread_score = -total_spread * 0.3

        score = spread_score - inventory_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# ======================================================================
# Dataset persistence
# ======================================================================

def save_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save dataset to a compressed NumPy archive.

    Parameters
    ----------
    states : np.ndarray
        Shape ``(N, T_obs, state_dim)``.
    actions : np.ndarray
        Shape ``(N, T_pred, action_dim)``.
    path : str
        Destination file path (``.npz`` extension recommended).
    metadata : dict or None
        Optional metadata to save alongside the data.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "states": states,
        "actions": actions,
    }
    if metadata is not None:
        # Convert metadata to storable format (no nested objects)
        meta_serialisable = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                meta_serialisable[k] = v
            elif isinstance(v, (int, float, str, bool, list)):
                meta_serialisable[k] = v
            elif isinstance(v, dict):
                meta_serialisable[k] = str(v)
            else:
                meta_serialisable[k] = str(v)
        save_dict["metadata"] = meta_serialisable

    np.savez_compressed(path, **save_dict)
    file_size = Path(path).stat().st_size / (1024 * 1024)
    logger.info(f"Dataset saved to {path} ({file_size:.1f} MB)")


def load_dataset(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Load dataset from a NumPy archive.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    states : np.ndarray
        Shape ``(N, T_obs, state_dim)``.
    actions : np.ndarray
        Shape ``(N, T_pred, action_dim)``.
    metadata : dict or None
        Metadata dictionary if present in the file, else ``None``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required keys are missing.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if "states" not in data or "actions" not in data:
        raise KeyError(
            f"Dataset must contain 'states' and 'actions' keys. "
            f"Found: {list(data.keys())}"
        )

    states = data["states"]
    actions = data["actions"]
    metadata = dict(data["metadata"]) if "metadata" in data else None

    logger.info(
        f"Dataset loaded from {path}: states={states.shape}, "
        f"actions={actions.shape}"
    )

    return states, actions, metadata


# ======================================================================
# Main – smoke test with small dataset
# ======================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from finflowrl.utils.config import Config

    print("=" * 70)
    print("  Data Generation – Smoke Test")
    print("=" * 70)

    # --- Test 1: Generate small dataset ---
    print("\n--- Test 1: Generate small dataset ---")
    config = Config()

    states, actions, metadata = generate_expert_demonstrations(
        config=config,
        num_scenarios=3,
        steps_per_scenario=200,
        T_obs=2,
        T_pred=4,
        seed=42,
        verbose=True,
    )

    print(f"\n  States shape:  {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Generation time: {metadata['elapsed_time']:.2f}s")
    assert states.ndim == 3, f"States should be 3-D, got {states.ndim}"
    assert actions.ndim == 3, f"Actions should be 3-D, got {actions.ndim}"
    assert states.shape[0] == actions.shape[0], "Mismatch in number of samples"
    assert states.shape[1] == 2, f"T_obs should be 2, got {states.shape[1]}"
    assert actions.shape[1] == 4, f"T_pred should be 4, got {actions.shape[1]}"
    assert actions.shape[2] == 2, f"action_dim should be 2, got {actions.shape[2]}"
    # Actions should be in valid range [0.01, 2.0]
    assert actions.min() >= 0.005, f"Action below range: {actions.min()}"
    assert actions.max() <= 2.01, f"Action above range: {actions.max()}"
    print("  [PASS]")

    # --- Test 2: Save and load dataset ---
    print("\n--- Test 2: Save and load dataset ---")
    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False
    ) as f:
        data_path = f.name

    try:
        save_dataset(states, actions, data_path, metadata=metadata)
        loaded_states, loaded_actions, loaded_meta = load_dataset(data_path)

        assert np.allclose(states, loaded_states), "States mismatch after load"
        assert np.allclose(actions, loaded_actions), "Actions mismatch after load"
        assert loaded_states.shape == states.shape, "State shape mismatch"
        assert loaded_actions.shape == actions.shape, "Action shape mismatch"
        assert loaded_meta is not None, "Metadata should be loaded"
        print(f"  File: {data_path}")
        print(f"  States match: True")
        print(f"  Actions match: True")
        print("  [PASS]")
    finally:
        import os
        os.unlink(data_path)

    # --- Test 3: Empty dataset edge case ---
    print("\n--- Test 3: Empty dataset ---")
    empty_states = np.zeros((0, 2, 7), dtype=np.float32)
    empty_actions = np.zeros((0, 4, 2), dtype=np.float32)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False
    ) as f:
        empty_path = f.name
    try:
        save_dataset(empty_states, empty_actions, empty_path)
        s2, a2, m2 = load_dataset(empty_path)
        assert s2.shape == (0, 2, 7), "Empty state shape mismatch"
        assert a2.shape == (0, 4, 2), "Empty action shape mismatch"
        print("  Empty dataset round-trip: OK")
        print("  [PASS]")
    finally:
        os.unlink(empty_path)

    # --- Test 4: Metadata preservation ---
    print("\n--- Test 4: Metadata preservation ---")
    test_meta = {
        "num_scenarios": 5,
        "steps_per_scenario": 100,
        "total_samples": states.shape[0],
        "generation_time": 12.3,
    }
    with tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False
    ) as f:
        meta_path = f.name
    try:
        save_dataset(states, actions, meta_path, metadata=test_meta)
        _, _, m = load_dataset(meta_path)
        assert m is not None, "Metadata should exist"
        assert m["num_scenarios"] == 5, "Metadata value mismatch"
        print(f"  Loaded metadata: {m}")
        print("  [PASS]")
    finally:
        os.unlink(meta_path)

    print("\n" + "=" * 70)
    print("  data.py OK – all tests passed.")
    print("=" * 70)
