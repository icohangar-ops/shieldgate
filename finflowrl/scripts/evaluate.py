"""CLI entry point for FinFlowRL evaluation and comparison."""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="FinFlowRL: Evaluate strategies on synthetic market data",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["random", "as", "glft", "glft-drift"],
        help="Strategies to evaluate",
    )
    args = parser.parse_args()

    from finflowrl.utils import Config, load_config
    from finflowrl.env import MarketSimulator, HFTEnv
    from finflowrl.experts import (
        AvellanedaStoikovExpert,
        GLFTExpert,
        GLFTDriftExpert,
    )
    from finflowrl.evaluation.metrics import evaluate_strategy, compare_strategies

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = Config()

    print("=" * 60)
    print("  FinFlowRL Strategy Evaluation")
    print("=" * 60)
    print(f"  Episodes:    {args.episodes}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Strategies:  {args.strategies}")
    print(f"  Market sigma: {config.market.sigma}")
    print("=" * 60)

    # Build expert dict
    experts = {
        "as": AvellanedaStoikovExpert(
            gamma=config.expert.as_gamma,
            k=config.expert.as_k,
        ),
        "glft": GLFTExpert(
            gamma=config.expert.glft_gamma,
            kappa=config.expert.glft_kappa,
        ),
        "glft-drift": GLFTDriftExpert(
            gamma=config.expert.glft_gamma,
            kappa=config.expert.glft_kappa,
        ),
    }

    results = []

    for strategy_name in args.strategies:
        all_returns = []
        for ep in range(args.episodes):
            sim = MarketSimulator(
                sigma=config.market.sigma,
                H=config.market.H,
                seed=args.seed + ep,
            )
            env = HFTEnv(sim, max_steps=args.max_steps)
            obs = env.reset()
            episode_returns = []

            done = False
            while not done:
                if strategy_name == "random":
                    action = np.random.uniform(0.01, 2.0, size=2)
                elif strategy_name in experts:
                    # Convert obs to MarketState for expert
                    from finflowrl.experts.base import MarketState
                    n_feat = env._n_features  # 7
                    latest = obs[-n_feat:]
                    state = MarketState(
                        mid_price=float(latest[0]),
                        inventory=int(latest[1]),
                        spread=float(latest[2]),
                        buy_intensity=float(latest[3]),
                        sell_intensity=float(latest[4]),
                        price_change=float(latest[5]),
                        volatility=float(latest[6]) if latest[6] > 0 else 0.01,
                        time_remaining=max(1.0 - env._current_step / env.max_steps, 0.0),
                    )
                    expert_action = experts[strategy_name].act(state)
                    action = np.array([expert_action.delta_bid, expert_action.delta_ask])
                else:
                    action = np.array([0.1, 0.1])

                obs, reward, done, info = env.step(action)
                episode_returns.append(reward)

            all_returns.extend(episode_returns)

        result = evaluate_strategy(
            np.array(all_returns),
            strategy_name=strategy_name.upper(),
            market_condition=f"sigma={config.market.sigma}",
        )
        results.append(result)
        print(f"  [{ep+1}/{args.episodes}] {result.summary()}")

    print("\n" + compare_strategies(results))
    print("[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
