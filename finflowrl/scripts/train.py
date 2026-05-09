"""CLI entry point for FinFlowRL training pipeline."""

import argparse
import sys

from finflowrl.utils import Config, load_config


def main():
    parser = argparse.ArgumentParser(
        description="FinFlowRL: Train Imitation-RL policies for financial stochastic control",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--stage", type=str, choices=["pretrain", "finetune", "both"], default="pretrain",
        help="Training stage: pretrain (Stage 1), finetune (Stage 2), or both",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override compute device (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pre-trained checkpoint for fine-tuning",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"[WARNING] Config file '{args.config}' not found. Using defaults.")
        config = Config()

    # Apply CLI overrides
    if args.epochs is not None:
        if args.stage in ("pretrain", "both"):
            config.pretrain.num_epochs = args.epochs
        if args.stage in ("finetune", "both"):
            config.finetune.ppo_epochs = args.epochs
    if args.batch_size is not None:
        config.pretrain.batch_size = args.batch_size
    if args.lr is not None:
        config.pretrain.learning_rate = args.lr
        config.finetune.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    print("=" * 60)
    print("  FinFlowRL Training Pipeline")
    print("=" * 60)
    print(config.summary())
    print(f"  Stage:       {args.stage}")
    print(f"  Checkpoint:  {args.checkpoint or 'None'}")
    print("=" * 60)

    # Resolve device
    if config.device == "auto":
        import torch
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {config.device}")

    # Stage 1: Pre-train MeanFlow on expert demonstrations
    if args.stage in ("pretrain", "both"):
        print("\n[Stage 1] MeanFlow Pre-training")
        print("-" * 40)

        from finflowrl.models import MeanFlowPolicy
        from finflowrl.training import MeanFlowPretrainer
        from finflowrl.utils.data import generate_expert_demonstrations

        print("  Generating expert demonstrations...")
        states, actions, meta = generate_expert_demonstrations(
            config, num_scenarios=4, steps_per_scenario=100
        )
        print(f"  Dataset: {len(states)} samples")

        model = MeanFlowPolicy(
            state_dim=config.model.state_dim,
            action_dim=config.model.action_dim,
            noise_dim=config.model.noise_dim,
            T_obs=config.model.T_obs,
            T_pred=config.model.T_pred,
            T_exec=config.model.T_exec,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
        )
        print(f"  Model parameters: {model.count_parameters():,}")

        pretrainer = MeanFlowPretrainer(
            model=model,
            learning_rate=config.pretrain.learning_rate,
            weight_decay=config.pretrain.weight_decay,
            batch_size=config.pretrain.batch_size,
            num_epochs=config.pretrain.num_epochs,
            device=config.device,
        )
        history = pretrainer.train(
            states=states,
            actions=actions,
        )
        print(f"  Final loss: {history['loss'][-1]:.6f}")
        print("  [Stage 1 Complete]")

    # Stage 2: Fine-tune with PPO in noise space
    if args.stage in ("finetune", "both"):
        print("\n[Stage 2] FlowRL Fine-tuning")
        print("-" * 40)

        from finflowrl.models import MeanFlowPolicy, NoisePolicy
        from finflowrl.training import FlowRLFinetuner

        model = MeanFlowPolicy(
            state_dim=config.model.state_dim,
            action_dim=config.model.action_dim,
            noise_dim=config.model.noise_dim,
            T_obs=config.model.T_obs,
            T_pred=config.model.T_pred,
            T_exec=config.model.T_exec,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
        )

        if args.checkpoint:
            print(f"  Loading checkpoint: {args.checkpoint}")
            pretrainer = MeanFlowPretrainer(model=model, device=config.device)
            pretrainer.load_checkpoint(args.checkpoint)

        noise_policy = NoisePolicy(
            state_dim=config.model.T_obs * config.model.state_dim,
            noise_dim=config.model.T_pred * config.model.noise_dim,
            hidden_dim=64,
        )
        print(f"  NoisePolicy parameters: {noise_policy.count_parameters()::,}")

        finetuner = FlowRLFinetuner(
            meanflow_model=model,
            noise_policy=noise_policy,
            learning_rate=config.finetune.learning_rate,
            gamma=config.finetune.gamma,
            clip_epsilon=config.finetune.clip_epsilon,
            device=config.device,
        )
        history = finetuner.train(total_timesteps=1000)
        print(f"  Final reward: {history['reward'][-1]:.4f}")
        print("  [Stage 2 Complete]")

    print("\n[Done] Training complete.")


if __name__ == "__main__":
    main()
