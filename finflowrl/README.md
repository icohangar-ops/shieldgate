# FinFlowRL

**Imitation-Reinforcement Learning Framework for Adaptive Stochastic Control in Finance**

Based on: Li et al. (2025) "FinFlowRL" — [arXiv:2509.17964](https://arxiv.org/abs/2509.17964)

## Overview

FinFlowRL combines imitation learning with reinforcement learning for financial stochastic control. The framework first pre-trains a MeanFlow policy by learning from multiple expert strategies, then fine-tunes it through PPO in noise space — achieving higher Sharpe ratios and lower maximum drawdowns than traditional methods.

### Key Innovations

- **MeanFlow Pre-training**: Flow matching in average velocity space (Geng et al. 2025) for one-step action generation
- **FlowRL Fine-tuning**: PPO in noise space with frozen expert — 84% fewer trainable parameters
- **Action Chunking**: Generates sequences of actions (T_pred=8, execute T_exec=4) to address non-Markovian market dynamics
- **FiLM Conditioning**: State-dependent action generation via Feature-wise Linear Modulation

### Application: High-Frequency Market-Making

The framework is applied to optimal market-making with:
- Jump-diffusion price process (Merton 1976) with fractional Brownian motion
- Bivariate Hawkes process for self/cross-exciting order arrivals
- Parimutuel-style execution with inventory risk penalty

## Architecture

```
Stage 1: MeanFlow Pre-training
  Expert Demos (AS, GLFT, GLFT-Drift, PPO)
    → Flow Matching (MeanFlow Consistency Loss)
      → Adaptive Meta-Policy g_θ

Stage 2: FlowRL Fine-tuning
  Frozen g_θ + Learnable Noise Policy π_φ
    → PPO in Noise Space
      → Optimized Adaptive Policy
```

## Installation

```bash
pip install -e ".[dev]"
```

Requires: Python >=3.10, PyTorch >=2.0, NumPy >=1.24, PyYAML >=6.0

## Quick Start

```python
import torch
from finflowrl.models import MeanFlowPolicy

# Create model
model = MeanFlowPolicy(
    state_dim=14, action_dim=2, noise_dim=16,
    T_obs=2, T_pred=8, T_exec=4,
    hidden_dim=128, num_layers=3,
)

# Generate actions from market observations
states = torch.randn(1, 2, 14)  # (batch, T_obs, state_dim)
actions = model.generate(states)   # (1, T_pred, action_dim)
print(f"Action chunk: {actions.shape}, range [{actions.min():.3f}, {actions.max():.3f}]")
```

```python
from finflowrl.env import MarketSimulator, HFTEnv
from finflowrl.experts import AvellanedaStoikovExpert
from finflowrl.evaluation import evaluate_strategy

# Run a simulation
sim = MarketSimulator(sigma=0.1, seed=42)
env = HFTEnv(sim, max_steps=500)
expert = AvellanedaStoikovExpert()

obs = env.reset()
returns = []
for _ in range(500):
    # Expert action (in practice, use model.generate())
    action = [0.1, 0.1]
    obs, reward, done, info = env.step(action)
    returns.append(reward)
    if done:
        break

result = evaluate_strategy(returns, "MyStrategy")
print(result.summary())
```

## CLI

```bash
# Stage 1: Pre-train on expert demonstrations
python -m scripts.train --stage pretrain --epochs 100

# Stage 2: Fine-tune with PPO
python -m scripts.train --stage finetune --checkpoint checkpoints/pretrain/best.pt

# Evaluate strategies
python -m scripts.evaluate --strategies random as glft glft-drift --episodes 10

# Run all tests
pytest tests/ -v
```

## Project Structure

```
finflowrl/
├── env/                    # Market simulation
│   ├── market_simulator.py # Jump-diffusion + Hawkes process
│   └── hft_env.py          # Gym-style HFT environment
├── experts/                # Expert strategies
│   ├── base.py             # Expert ABC + MarketState
│   ├── avellaneda_stoikov.py
│   ├── glft.py
│   ├── glft_drift.py
│   └── ppo_expert.py       # Numpy MLP expert
├── models/                 # Neural networks
│   ├── film.py             # FiLM conditioning layer
│   ├── meanflow.py         # MeanFlow policy (core)
│   └── noise_policy.py     # Gaussian noise policy for PPO
├── training/               # Training pipelines
│   ├── pretrain.py         # Stage 1: MeanFlow pre-training
│   └── finetune.py         # Stage 2: FlowRL fine-tuning
├── evaluation/             # Metrics & comparison
│   └── metrics.py          # PnL, Sharpe Ratio, MDD
├── utils/                  # Configuration & data
│   ├── config.py           # YAML config system
│   └── data.py             # Expert demo generation
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py            # CLI training entry point
│   └── evaluate.py         # CLI evaluation entry point
└── tests/                  # Unit tests
```

## Evaluation Metrics

| Method | PnL ↑ | Sharpe Ratio ↑ | Max Drawdown ↓ |
|--------|-------|----------------|-----------------|
| Random Action | 1.99 | 0.06 | 28.49% |
| AS | 24.22 | 0.09 | 241.65% |
| GLFT | 25.10 | 0.37 | 60.57% |
| Vanilla PPO | 14.76 | 0.10 | 133.61% |
| Pretrained MeanFlow | 23.91 | 0.37 | 43.40% |
| **FinFlowRL** | **26.33** | **0.50** | **45.47%** |

Results from the paper (1M trials, various market conditions).

## Citation

```bibtex
@article{li2025finflowrl,
  title={FinFlowRL: An Imitation-Reinforcement Learning Framework for Adaptive Stochastic Control in Finance},
  author={Li, Yang and Chen, Zhi and Yang, Steve Y. and Zhang, Ruixun},
  journal={arXiv preprint arXiv:2509.17964},
  year={2025}
}
```

## License

MIT
