"""Hierarchical configuration for FinFlowRL experiments.

Uses plain Python dataclasses for type safety and IDE autocomplete, with
YAML serialisation for human-readable experiment configs.  All sections
map directly to the components of the FinFlowRL pipeline:

- :class:`MarketConfig` – Market simulator parameters (jump-diffusion,
  Hawkes process, etc.)
- :class:`ExpertConfig` – Expert strategy hyperparameters.
- :class:`ModelConfig` – Neural network architecture settings.
- :class:`PretrainConfig` – Stage-1 (MeanFlow) training settings.
- :class:`FinetuneConfig` – Stage-2 (FlowRL / PPO) training settings.
- :class:`Config` – Top-level container combining all sections.

Usage
-----
    >>> config = Config.from_yaml("configs/default.yaml")
    >>> config.market.sigma = 0.2  # programmatic override
    >>> config.to_yaml("configs/modified.yaml")

Reference
---------
    FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive
    Stochastic Control in Finance (arXiv 2509.17964)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ======================================================================
# Market simulator configuration
# ======================================================================

@dataclass
class MarketConfig:
    """Market simulator parameters.

    Controls the jump-diffusion price process and bivariate Hawkes order
    arrival process used to generate synthetic market data.

    Attributes mirror the constructor of :class:`MarketSimulator`.
    """

    # --- Price process ---
    S0: float = 100.0          # Initial mid-price
    mu: float = 0.0            # Drift rate
    sigma: float = 0.1         # Diffusion volatility
    H: float = 0.5             # Hurst exponent (0.5 = standard BM)
    mu_J: float = -0.02        # Jump mean
    sigma_J: float = 0.03      # Jump std-dev
    lambda_J: float = 0.1      # Jump intensity (Poisson rate)
    dt: float = 0.01           # Time-step size

    # --- Hawkes order-arrival process ---
    mu_a: float = 10.0         # Baseline buy-order arrival rate
    mu_b: float = 10.0         # Baseline sell-order arrival rate
    alpha_aa: float = 5.0      # Buy self-excitation
    alpha_ab: float = 3.0      # Buy cross-excitation (from sell)
    alpha_bb: float = 5.0      # Sell self-excitation
    alpha_ba: float = 3.0      # Sell cross-excitation (from buy)
    beta: float = 10.0         # Exponential decay rate


# ======================================================================
# Expert strategy configuration
# ======================================================================

@dataclass
class ExpertConfig:
    """Expert strategy hyperparameters.

    Controls the Avellaneda-Stoikov, GLFT, GLFT-Drift, and PPO expert
    strategies used for generating imitation-learning demonstrations.
    """

    # --- Avellaneda-Stoikov ---
    as_gamma: float = 0.1      # Risk aversion
    as_k: float = 1.5          # Arrival intensity parameter
    as_sigma: float = 0.1      # Volatility estimate (None = use MarketState)
    as_T: float = 1.0          # Trading horizon

    # --- GLFT / GLFT-Drift ---
    glft_gamma: float = 0.1    # Risk aversion
    glft_kappa: float = 1.5    # Intensity scaling parameter


# ======================================================================
# Neural network architecture
# ======================================================================

@dataclass
class ModelConfig:
    """Neural network architecture settings.

    Controls the MeanFlow policy, FiLM conditioning, and noise policy
    dimensions.
    """

    state_dim: int = 14        # Per-timestep state features (T_obs * 7)
    action_dim: int = 2        # (delta_bid, delta_ask)
    noise_dim: int = 16        # Per-timestep latent noise dimension
    hidden_dim: int = 128      # Hidden layer width
    num_layers: int = 3        # Number of FiLM residual blocks
    T_obs: int = 2             # Observation window length
    T_pred: int = 8            # Action prediction horizon (chunk size)
    T_exec: int = 4            # Actions to execute before re-planning


# ======================================================================
# Stage 1: Pre-training
# ======================================================================

@dataclass
class PretrainConfig:
    """Stage-1 MeanFlow pre-training settings."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 100
    warmup_steps: int = 1000


# ======================================================================
# Stage 2: Fine-tuning
# ======================================================================

@dataclass
class FinetuneConfig:
    """Stage-2 FlowRL fine-tuning settings (PPO in noise space)."""

    learning_rate: float = 3e-4
    gamma: float = 0.99        # PPO discount factor
    clip_epsilon: float = 0.2  # PPO clip parameter
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gae_lambda: float = 0.95
    rollout_steps: int = 2048
    ppo_epochs: int = 10


# ======================================================================
# Top-level config
# ======================================================================

@dataclass
class Config:
    """Top-level experiment configuration.

    Combines all sub-configs and global settings.  Supports round-trip
    serialisation to/from YAML.

    Parameters
    ----------
    market : MarketConfig
        Market simulator parameters.
    expert : ExpertConfig
        Expert strategy hyperparameters.
    model : ModelConfig
        Neural network architecture.
    pretrain : PretrainConfig
        Stage-1 pre-training settings.
    finetune : FinetuneConfig
        Stage-2 fine-tuning settings.
    seed : int
        Global random seed.  Default ``42``.
    device : str
        Compute device (``"auto"``, ``"cpu"``, ``"cuda"``).
    checkpoint_dir : str
        Root directory for checkpoints.
    data_dir : str
        Root directory for datasets.
    """

    market: MarketConfig = field(default_factory=MarketConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (YAML-serialisable).

        Returns
        -------
        dict
            Nested dictionary of all config values.
        """
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create a Config from a nested dictionary.

        Sub-configs are created from their respective dictionaries if
        present; otherwise defaults are used.  Unknown keys are ignored.

        Parameters
        ----------
        d : dict
            Configuration dictionary (e.g. from YAML loading).

        Returns
        -------
        Config
        """
        market_d = d.get("market", {})
        expert_d = d.get("expert", {})
        model_d = d.get("model", {})
        pretrain_d = d.get("pretrain", {})
        finetune_d = d.get("finetune", {})

        return cls(
            market=MarketConfig(**{k: v for k, v in market_d.items() if k in MarketConfig.__dataclass_fields__}),
            expert=ExpertConfig(**{k: v for k, v in expert_d.items() if k in ExpertConfig.__dataclass_fields__}),
            model=ModelConfig(**{k: v for k, v in model_d.items() if k in ModelConfig.__dataclass_fields__}),
            pretrain=PretrainConfig(**{k: v for k, v in pretrain_d.items() if k in PretrainConfig.__dataclass_fields__}),
            finetune=FinetuneConfig(**{k: v for k, v in finetune_d.items() if k in FinetuneConfig.__dataclass_fields__}),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
            checkpoint_dir=d.get("checkpoint_dir", "checkpoints"),
            data_dir=d.get("data_dir", "data"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str
            Source YAML file path.

        Returns
        -------
        Config

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            d = yaml.safe_load(f)
        if d is None:
            d = {}
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "FinFlowRL Configuration",
            "=" * 50,
            f"  Seed:            {self.seed}",
            f"  Device:          {self.device}",
            f"  Checkpoint dir:  {self.checkpoint_dir}",
            f"  Data dir:        {self.data_dir}",
            "",
            "  Market:",
            f"    S0={self.market.S0}, sigma={self.market.sigma}, "
            f"H={self.market.H}, dt={self.market.dt}",
            f"    mu_a={self.market.mu_a}, mu_b={self.market.mu_b}, "
            f"beta={self.market.beta}",
            "",
            "  Model:",
            f"    state_dim={self.model.state_dim}, "
            f"action_dim={self.model.action_dim}, "
            f"noise_dim={self.model.noise_dim}",
            f"    hidden_dim={self.model.hidden_dim}, "
            f"num_layers={self.model.num_layers}",
            f"    T_obs={self.model.T_obs}, T_pred={self.model.T_pred}, "
            f"T_exec={self.model.T_exec}",
            "",
            "  Pretrain:",
            f"    lr={self.pretrain.learning_rate}, "
            f"batch={self.pretrain.batch_size}, "
            f"epochs={self.pretrain.num_epochs}",
            "",
            "  Finetune:",
            f"    lr={self.finetune.learning_rate}, "
            f"gamma={self.finetune.gamma}, "
            f"clip={self.finetune.clip_epsilon}",
            f"    rollout_steps={self.finetune.rollout_steps}, "
            f"ppo_epochs={self.finetune.ppo_epochs}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ======================================================================
# Free-standing helper
# ======================================================================

def load_config(path: str) -> Config:
    """Convenience function to load a Config from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
    """
    return Config.from_yaml(path)


# ======================================================================
# Main – smoke test
# ======================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  Config – Smoke Test")
    print("=" * 60)

    # --- Test 1: Default construction ---
    print("\n--- Test 1: Default config ---")
    config = Config()
    print(config.summary())
    assert config.market.sigma == 0.1
    assert config.model.state_dim == 14
    assert config.seed == 42
    print("  [PASS]")

    # --- Test 2: Round-trip YAML serialisation ---
    print("\n--- Test 2: YAML round-trip ---")
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml_path = f.name

    try:
        config.to_yaml(yaml_path)
        loaded = Config.from_yaml(yaml_path)

        assert loaded.market.sigma == config.market.sigma
        assert loaded.model.T_pred == config.model.T_pred
        assert loaded.finetune.gamma == config.finetune.gamma
        assert loaded.seed == config.seed

        print(f"  Saved to: {yaml_path}")
        print(f"  Loaded config seed: {loaded.seed}")
        print(f"  Loaded market sigma: {loaded.market.sigma}")
        print("  [PASS]")
    finally:
        import os
        os.unlink(yaml_path)

    # --- Test 3: from_dict with overrides ---
    print("\n--- Test 3: from_dict with overrides ---")
    override = Config.from_dict({
        "market": {"sigma": 0.3, "H": 0.7},
        "model": {"hidden_dim": 256, "T_pred": 16},
        "seed": 123,
    })
    assert override.market.sigma == 0.3
    assert override.market.H == 0.7
    assert override.model.hidden_dim == 256
    assert override.model.T_pred == 16
    assert override.seed == 123
    # Non-overridden values stay default
    assert override.market.S0 == 100.0
    assert override.model.state_dim == 14
    print(f"  Override market sigma: {override.market.sigma}")
    print(f"  Override model hidden: {override.model.hidden_dim}")
    print("  [PASS]")

    # --- Test 4: to_dict ---
    print("\n--- Test 4: to_dict ---")
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "market" in d
    assert "model" in d
    assert d["market"]["sigma"] == 0.1
    print(f"  Dict keys: {list(d.keys())}")
    print("  [PASS]")

    # --- Test 5: load_config helper ---
    print("\n--- Test 5: load_config helper ---")
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as f:
        yaml_path2 = f.name
        yaml.dump(config.to_dict(), f)

    try:
        loaded2 = load_config(yaml_path2)
        assert loaded2.seed == config.seed
        print("  load_config() works correctly")
        print("  [PASS]")
    finally:
        os.unlink(yaml_path2)

    print("\n" + "=" * 60)
    print("  config.py OK – all tests passed.")
    print("=" * 60)
