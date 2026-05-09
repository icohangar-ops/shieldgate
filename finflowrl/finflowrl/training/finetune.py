"""Stage 2: FlowRL fine-tuning with PPO in noise space.

Implements the FlowRL fine-tuning algorithm that dramatically reduces
trainable parameters (~84%) by keeping the pre-trained MeanFlow model
frozen and training only a lightweight Gaussian noise policy via PPO.

Action Generation Pipeline
---------------------------
    1. Observe market state s_t
    2. Sample noise w ~ π_φ(w | s_t)  (trainable noise policy)
    3. Generate action a = g_θ(s_t, w)  (frozen MeanFlow, one-step):
       - z_1 ~ N(0, I)
       - u_θ = MeanFlow.predict_velocity(z_1 + w, t=1, r=0, s_t)
       - a_latent = z_1 + w − u_θ
       - a = MeanFlow.decode_actions(a_latent)
    4. Execute first action of chunk in environment
    5. Collect reward, advance state

PPO Algorithm
-------------
    - Collect rollout of transitions (s, w, log_prob, reward, value)
    - Compute GAE (Generalised Advantage Estimation) for variance reduction
    - Update noise policy via clipped surrogate objective:
        L = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
    - Joint optimisation with entropy bonus and value loss

References
----------
    - FinFlowRL (arXiv 2509.17964)
    - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Value Network
# ======================================================================

class ValueNetwork(nn.Module):
    """Simple MLP value function baseline for PPO.

    Estimates V(s) for Generalised Advantage Estimation.

    Parameters
    ----------
    state_dim : int
        Input observation dimension.
    hidden_dim : int
        Hidden layer width.  Default ``64``.
    num_layers : int
        Number of hidden layers.  Default ``2``.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_d = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.SiLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialisation."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V(s).

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.
        """
        return self.net(state)


# ======================================================================
# Rollout Buffer
# ======================================================================

class _RolloutBuffer:
    """Fixed-size buffer for storing rollout transitions.

    Stores (state, noise, log_prob, reward, value, done) tuples and
    provides methods for computing returns and GAE advantages.
    """

    def __init__(self, buffer_size: int, state_dim: int, noise_dim: int) -> None:
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.ptr: int = 0
        self.full: bool = False

        self.states = np.zeros(
            (buffer_size, state_dim), dtype=np.float32
        )
        self.noises = np.zeros(
            (buffer_size, noise_dim), dtype=np.float32
        )
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def reset(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.full = False
        self.states[:] = 0.0
        self.noises[:] = 0.0
        self.log_probs[:] = 0.0
        self.rewards[:] = 0.0
        self.values[:] = 0.0
        self.dones[:] = 0.0
        self.advantages[:] = 0.0
        self.returns[:] = 0.0

    def push(
        self,
        state: np.ndarray,
        noise: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a single transition.

        Parameters
        ----------
        state : np.ndarray
            Shape ``(state_dim,)``.
        noise : np.ndarray
            Shape ``(noise_dim,)``.
        log_prob : float
            Log probability of the noise sample.
        reward : float
            Environment reward.
        value : float
            Value function estimate V(s).
        done : bool
            Whether the episode ended.
        """
        if self.ptr >= self.buffer_size:
            self.full = True
            return
        self.states[self.ptr] = state
        self.noises[self.ptr] = noise
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self.ptr


# ======================================================================
# FlowRL Fine-tuner
# ======================================================================

class FlowRLFinetuner:
    """Stage 2: Fine-tune via PPO in noise space.

    Freezes the pre-trained MeanFlow model and trains a lightweight
    noise policy π_φ(w|s) that generates optimal input noise for
    action generation:

        w ~ π_φ(w | s),  a = g_θ(s, w)

    where g_θ is the frozen MeanFlow policy.

    Parameters
    ----------
    meanflow_model : nn.Module
        A pre-trained :class:`MeanFlowPolicy` (will be frozen).
    noise_policy : nn.Module
        A trainable :class:`NoisePolicy`.
    env : object
        An :class:`HFTEnv` environment instance.
    learning_rate : float
        Learning rate for the noise policy and value network.
        Default ``3e-4``.
    gamma : float
        PPO discount factor.  Default ``0.99``.
    clip_epsilon : float
        PPO clipping parameter.  Default ``0.2``.
    entropy_coeff : float
        Entropy bonus coefficient to encourage exploration.
        Default ``0.01``.
    value_coeff : float
        Value loss coefficient.  Default ``0.5``.
    gae_lambda : float
        GAE lambda parameter for advantage estimation.
        Default ``0.95``.
    batch_size : int
        Mini-batch size for PPO updates.  Default ``64``.
    num_epochs : int
        Number of PPO optimisation epochs per rollout.
        Default ``10``.
    rollout_steps : int
        Number of environment steps per rollout collection.
        Default ``2048``.
    checkpoint_dir : str
        Directory to save checkpoints.  Default ``"checkpoints/finetune"``.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  Default ``"auto"``.
    grad_clip : float
        Maximum gradient norm for clipping.  Set to ``0.0`` to disable.
        Default ``0.5``.
    log_interval : int
        Log metrics every N rollout steps.  Default ``500``.
    """

    def __init__(
        self,
        meanflow_model: nn.Module,
        noise_policy: nn.Module,
        env: Any,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        gae_lambda: float = 0.95,
        batch_size: int = 64,
        num_epochs: int = 10,
        rollout_steps: int = 2048,
        checkpoint_dir: str = "checkpoints/finetune",
        device: str = "auto",
        grad_clip: float = 0.5,
        log_interval: int = 500,
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --- Freeze MeanFlow model ---
        self.meanflow_model = meanflow_model.to(self.device)
        self.meanflow_model.eval()
        for param in self.meanflow_model.parameters():
            param.requires_grad = False
        logger.info("MeanFlow model frozen for fine-tuning.")

        # --- Trainable noise policy ---
        self.noise_policy = noise_policy.to(self.device)
        self.noise_dim = noise_policy.noise_dim
        logger.info(
            f"Noise policy params: "
            f"{sum(p.numel() for p in noise_policy.parameters()):,}"
        )

        # --- Environment ---
        self.env = env
        self.state_dim = env.obs_dim

        # --- Value network ---
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=64,
            num_layers=2,
        ).to(self.device)

        # --- Hyperparameters ---
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.rollout_steps = rollout_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        # --- Optimisers (separate for policy and value) ---
        self.policy_optimizer = torch.optim.Adam(
            self.noise_policy.parameters(), lr=learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate,
        )

        # --- Training state ---
        self._total_timesteps: int = 0
        self._best_episode_reward: float = float("-inf")
        self._history: List[Dict[str, float]] = []

        # --- Rollout buffer ---
        self._buffer = _RolloutBuffer(
            buffer_size=rollout_steps,
            state_dim=self.state_dim,
            noise_dim=self.noise_dim,
        )

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Action generation with noise perturbation
    # ------------------------------------------------------------------

    def _generate_action_with_noise(
        self,
        obs: np.ndarray,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Generate action using frozen MeanFlow + noise perturbation.

        Implements: a = decode(z_1 + w − u_θ(z_1 + w, 0, 1, s))

        Parameters
        ----------
        obs : np.ndarray
            Environment observation, shape ``(state_dim,)``.
        noise : torch.Tensor
            Noise perturbation, shape ``(1, noise_dim)``.

        Returns
        -------
        torch.Tensor
            Action chunk, shape ``(1, T_pred, action_dim)``.
        """
        with torch.no_grad():
            # Reshape observation for model: (1, T_obs, state_dim)
            # The model expects 3-D states, but env gives flat (state_dim,)
            # We reshape assuming T_obs * single_step_features = state_dim
            T_obs = self.meanflow_model.T_obs
            single_dim = self.state_dim // T_obs
            state_3d = obs.reshape(1, T_obs, single_dim)

            state_3d = torch.FloatTensor(state_3d).to(self.device)

            # One-step generation with noise perturbation
            B = 1
            chunk_noise_dim = self.meanflow_model.chunk_noise_dim

            # z_1 ~ N(0, I) is already inside generate(),
            # so we apply noise after generation.
            # Instead, we manually replicate the generate() logic
            # to inject noise at the right point.
            z_1 = torch.randn(
                B, chunk_noise_dim, device=self.device, dtype=state_3d.dtype
            )

            # Inject noise perturbation
            z_1_perturbed = z_1 + noise  # (1, chunk_noise_dim)

            t = torch.ones(B, 1, device=self.device, dtype=state_3d.dtype)
            r = torch.zeros(B, 1, device=self.device, dtype=state_3d.dtype)

            state_feat = self.meanflow_model.encode_state(state_3d)
            u = self.meanflow_model.predict_velocity(z_1_perturbed, t, r, state_feat)
            a_latent = z_1_perturbed - u
            actions = self.meanflow_model.decode_actions(a_latent)

            return actions  # (1, T_pred, action_dim)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> Dict[str, Any]:
        """Collect trajectory data by running the noise policy in the env.

        For each step:
        1. Observe state s_t
        2. Sample noise w ~ π_φ(w|s_t)
        3. Generate action a = g_θ(s_t, w) (frozen MeanFlow)
        4. Execute first action of chunk in env
        5. Store (s_t, w, log_prob, reward, value, done)

        Returns
        -------
        dict
            Rollout statistics: ``episode_rewards``, ``mean_reward``,
            ``num_episodes``, ``rollout_length``.
        """
        self._buffer.reset()
        self.noise_policy.eval()

        episode_rewards: List[float] = []
        current_episode_reward: float = 0.0
        num_episodes: int = 0

        obs = self.env.reset()

        for step in range(self.rollout_steps):
            # Convert obs to tensor for policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Sample noise w ~ π_φ(w|s)
            with torch.no_grad():
                noise = self.noise_policy.sample(obs_tensor)
                log_prob = (
                    self.noise_policy.log_prob(noise, obs_tensor)
                    .sum(dim=-1)
                    .item()
                )
                value = self.value_net(obs_tensor).item()

            # Generate action with noise
            action_chunk = self._generate_action_with_noise(obs, noise)

            # Take first action from chunk
            action = action_chunk[0, 0, :].cpu().numpy()

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Store transition
            self._buffer.push(
                state=obs,
                noise=noise.cpu().numpy().flatten(),
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
            )

            current_episode_reward += reward
            self._total_timesteps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                if current_episode_reward > self._best_episode_reward:
                    self._best_episode_reward = current_episode_reward
                current_episode_reward = 0.0
                num_episodes += 1
                obs = self.env.reset()
            else:
                obs = next_obs

        # Handle any incomplete episode
        if current_episode_reward > 0:
            episode_rewards.append(current_episode_reward)

        rollout_stats = {
            "episode_rewards": episode_rewards,
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "num_episodes": num_episodes,
            "rollout_length": self._buffer.size,
            "total_timesteps": self._total_timesteps,
        }

        return rollout_stats

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalised Advantage Estimation.

        Uses TD(lambda) for bias-variance tradeoff:

            A_t = sum_{l=0}^{T-t} (γλ)^l δ_{t+l}

        where δ_t = r_t + γ V(s_{t+1}) − V(s_t).

        Parameters
        ----------
        rewards : np.ndarray
            1-D array of per-step rewards.
        values : np.ndarray
            1-D array of per-step value estimates V(s_t).
        dones : np.ndarray
            1-D array of done flags (float).
        last_value : float
            Bootstrap value for the last non-terminal state.
            Default ``0.0``.

        Returns
        -------
        advantages : np.ndarray
            GAE advantages, shape ``(T,)``.
        returns : np.ndarray
            Discounted returns, shape ``(T,)``.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD error: δ_t = r_t + γ V(s_{t+1}) − V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values

        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update(self) -> Dict[str, float]:
        """Run PPO update on the collected rollout data.

        For ``num_epochs`` iterations, samples mini-batches from the
        rollout buffer and updates the noise policy and value network.

        Loss = -L_clip + c_v * L_value - c_e * H[π]

        where:
        - L_clip = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
        - L_value = MSE(V(s), R_t)
        - H[π] = entropy of the noise policy

        Returns
        -------
        dict
            Update statistics: ``policy_loss``, ``value_loss``,
            ``entropy``, ``approx_kl``, ``clip_frac``.
        """
        # Compute GAE
        n = self._buffer.size
        if n == 0:
            return {}

        advantages, returns = self.compute_gae(
            rewards=self._buffer.rewards[:n],
            values=self._buffer.values[:n],
            dones=self._buffer.dones[:n],
        )

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors
        states_t = torch.FloatTensor(self._buffer.states[:n]).to(self.device)
        noises_t = torch.FloatTensor(self._buffer.noises[:n]).to(self.device)
        old_log_probs_t = torch.FloatTensor(
            self._buffer.log_probs[:n]
        ).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        self.noise_policy.train()
        self.value_net.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for epoch in range(self.num_epochs):
            # Generate random permutation for mini-batch sampling
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb_idx = indices[start:end]

                # Mini-batch data
                mb_states = states_t[mb_idx]
                mb_noises = noises_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # --- Policy loss ---
                # New log probabilities
                mb_new_log_probs = (
                    self.noise_policy.log_prob(mb_noises, mb_states)
                    .sum(dim=-1)
                )

                # Importance ratio
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)

                # Clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Entropy bonus ---
                entropy = self.noise_policy.entropy(mb_states).mean()

                # --- Value loss ---
                mb_values = self.value_net(mb_states).squeeze(-1)
                value_loss = F.mse_loss(mb_values, mb_returns)

                # --- Total loss ---
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                # --- Backprop (policy) ---
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.noise_policy.parameters())
                        + list(self.value_net.parameters()),
                        self.grad_clip,
                    )

                self.policy_optimizer.step()
                self.value_optimizer.step()

                # --- Logging ---
                with torch.no_grad():
                    approx_kl = (
                        (mb_old_log_probs - mb_new_log_probs).mean().item()
                    )
                    clip_frac = float(
                        (torch.abs(ratio - 1.0) > self.clip_epsilon)
                        .float()
                        .mean()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                n_updates += 1

        avg_metrics = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "clip_frac": total_clip_frac / max(n_updates, 1),
        }

        return avg_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_interval: int = 5000,
    ) -> Dict[str, Any]:
        """Full fine-tuning loop with periodic evaluation.

        Parameters
        ----------
        total_timesteps : int
            Total number of environment steps to train.
            Default ``100_000``.
        eval_interval : int
            Evaluate the policy (zero-noise) every N timesteps.
            Default ``5000``.

        Returns
        -------
        dict
            Training summary with ``history``, ``total_timesteps``,
            ``best_episode_reward``, and checkpoint paths.
        """
        logger.info(
            f"Starting FlowRL fine-tuning: "
            f"total_timesteps={total_timesteps}, "
            f"rollout_steps={self.rollout_steps}"
        )

        start_time = time.time()
        rollout_count = 0

        while self._total_timesteps < total_timesteps:
            rollout_count += 1

            # Collect rollout
            rollout_stats = self.collect_rollout()
            mean_reward = rollout_stats["mean_reward"]

            # PPO update
            update_metrics = self.ppo_update()

            # Combine metrics
            step_metrics = {
                "timesteps": self._total_timesteps,
                "mean_episode_reward": mean_reward,
                "std_episode_reward": rollout_stats.get("std_reward", 0.0),
                "num_episodes": rollout_stats["num_episodes"],
                **update_metrics,
            }
            self._history.append(step_metrics)

            # Logging
            if rollout_count % max(1, self.log_interval // self.rollout_steps) == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self._total_timesteps / max(elapsed, 1e-6)
                logger.info(
                    f"[Rollout {rollout_count:>4d}] "
                    f"timesteps={self._total_timesteps:>8d} | "
                    f"mean_reward={mean_reward:>8.2f} | "
                    f"policy_loss={update_metrics.get('policy_loss', 0):>8.4f} | "
                    f"value_loss={update_metrics.get('value_loss', 0):>8.4f} | "
                    f"entropy={update_metrics.get('entropy', 0):>6.4f} | "
                    f"({steps_per_sec:.0f} steps/s)"
                )

            # Periodic checkpointing
            if self._total_timesteps % eval_interval < self.rollout_steps:
                ckpt_path = str(
                    self.checkpoint_dir
                    / f"checkpoint_{self._total_timesteps}.pt"
                )
                self.save_checkpoint(ckpt_path)
                logger.info(
                    f"Checkpoint saved: {ckpt_path} "
                    f"(best_reward={self._best_episode_reward:.2f})"
                )

        # Final checkpoint
        final_path = str(self.checkpoint_dir / "final_model.pt")
        self.save_checkpoint(final_path)
        elapsed = time.time() - start_time

        logger.info(
            f"Fine-tuning complete: {self._total_timesteps} timesteps "
            f"in {elapsed:.1f}s "
            f"(best_reward={self._best_episode_reward:.2f})"
        )

        return {
            "history": self._history,
            "total_timesteps": self._total_timesteps,
            "best_episode_reward": self._best_episode_reward,
            "final_checkpoint": final_path,
            "training_time": elapsed,
            "num_rollouts": rollout_count,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save noise policy and value network to a checkpoint file.

        The frozen MeanFlow model is *not* saved (it has its own checkpoints
        from Stage 1).

        Parameters
        ----------
        path : str
            Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "noise_policy_state_dict": self.noise_policy.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "total_timesteps": self._total_timesteps,
            "best_episode_reward": self._best_episode_reward,
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load noise policy and value network from a checkpoint file.

        Parameters
        ----------
        path : str
            Source checkpoint file path.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.noise_policy.load_state_dict(checkpoint["noise_policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self._total_timesteps = checkpoint.get("total_timesteps", 0)
        self._best_episode_reward = checkpoint.get("best_episode_reward", float("-inf"))
        logger.info(
            f"Checkpoint loaded from {path} "
            f"(timesteps={self._total_timesteps}, "
            f"best_reward={self._best_episode_reward:.2f})"
        )


# ======================================================================
# Main – smoke test with synthetic environment
# ======================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from finflowrl.models import MeanFlowPolicy, NoisePolicy
    from finflowrl.env import MarketSimulator, HFTEnv

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  FlowRLFinetuner – Smoke Test")
    print("=" * 70)

    # --- Create environment ---
    sim = MarketSimulator(seed=42)
    env = HFTEnv(simulator=sim, max_steps=200, T_obs=2)
    print(f"Environment obs_dim: {env.obs_dim}")

    # --- Create models ---
    meanflow = MeanFlowPolicy(
        state_dim=7,  # per-timestep features from HFTEnv
        action_dim=2,
        noise_dim=4,
        T_obs=2,
        T_pred=4,
        T_exec=2,
        hidden_dim=32,
        num_layers=2,
    )
    print(f"MeanFlow params: {meanflow.count_parameters():,}")

    noise_policy = NoisePolicy(
        state_dim=env.obs_dim,
        noise_dim=meanflow.chunk_noise_dim,
        hidden_dim=32,
        num_layers=1,
    )
    print(f"NoisePolicy params: {noise_policy.count_parameters():,}")
    reduction_pct = (1.0 - noise_policy.count_parameters() / meanflow.count_parameters()) * 100
    print(f"Parameter reduction: {reduction_pct:.1f}%")

    # --- Test: single rollout collection ---
    print("\n--- Test: Collect rollout ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        finetuner = FlowRLFinetuner(
            meanflow_model=meanflow,
            noise_policy=noise_policy,
            env=env,
            learning_rate=1e-3,
            rollout_steps=100,
            num_epochs=2,
            batch_size=16,
            checkpoint_dir=tmpdir,
            device="cpu",
            log_interval=50,
        )

        rollout_stats = finetuner.collect_rollout()
        print(f"  Rollout stats: {rollout_stats}")
        assert rollout_stats["rollout_length"] > 0, "Should have collected data"
        assert rollout_stats["num_episodes"] > 0, "Should have completed episodes"
        print("  [PASS]")

        # --- Test: GAE computation ---
        print("\n--- Test: GAE computation ---")
        rewards = np.random.randn(50)
        values = np.random.randn(50) * 0.1
        dones = np.zeros(50)
        dones[25] = 1.0  # episode boundary

        adv, ret = finetuner.compute_gae(rewards, values, dones)
        assert len(adv) == 50, "Advantages should have same length as rewards"
        assert len(ret) == 50, "Returns should have same length as rewards"
        print(f"  Advantages: mean={adv.mean():.4f}, std={adv.std():.4f}")
        print(f"  Returns:     mean={ret.mean():.4f}, std={ret.std():.4f}")
        print("  [PASS]")

        # --- Test: PPO update ---
        print("\n--- Test: PPO update ---")
        update_metrics = finetuner.ppo_update()
        print(f"  Update metrics: {update_metrics}")
        assert "policy_loss" in update_metrics, "Should have policy_loss"
        assert "value_loss" in update_metrics, "Should have value_loss"
        assert "entropy" in update_metrics, "Should have entropy"
        print("  [PASS]")

        # --- Test: checkpoint save/load ---
        print("\n--- Test: Checkpoint save/load ---")
        ckpt_path = os.path.join(tmpdir, "test_ft_ckpt.pt")
        finetuner.save_checkpoint(ckpt_path)
        assert os.path.isfile(ckpt_path), "Checkpoint should exist"

        # Create new finetuner and load
        noise_policy2 = NoisePolicy(
            state_dim=env.obs_dim,
            noise_dim=meanflow.chunk_noise_dim,
            hidden_dim=32,
            num_layers=1,
        )
        finetuner2 = FlowRLFinetuner(
            meanflow_model=meanflow,
            noise_policy=noise_policy2,
            env=env,
            checkpoint_dir=tmpdir,
            device="cpu",
        )
        finetuner2.load_checkpoint(ckpt_path)

        # Verify noise policy weights match
        for (n1, p1), (n2, p2) in zip(
            noise_policy.named_parameters(), noise_policy2.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), f"Parameter {n1} mismatch"
        print("  [PASS]")

    print("\n" + "=" * 70)
    print("  finetune.py OK – all tests passed.")
    print("=" * 70)
