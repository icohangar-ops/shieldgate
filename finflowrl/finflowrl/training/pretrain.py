"""Stage 1: MeanFlow pre-training on expert demonstrations.

Trains a :class:`MeanFlowPolicy` (flow-matching model) to map Gaussian noise
to expert trading actions, conditioned on market state observations.  The
training procedure follows the MeanFlow consistency framework:

    1. Sample source noise z_0 ~ N(0, I)
    2. Encode expert actions a_expert → a_latent
    3. Sample flow times t ~ U(0, 1) and reference times r ~ U(0, t)
    4. Build straight-line interpolant z_t = (1-t)*z_0 + t*a_latent
    5. Predict average velocity u_θ(z_t, r, t, s)
    6. Compute MeanFlow consistency loss:
       L = ‖u_θ + (t−r)·d_t[u_θ] − v_t‖²
       where v_t = a_latent − z_0
    7. Back-propagate and update model parameters

Action Chunking
---------------
The model predicts a sequence of ``T_pred`` future actions per forward pass,
executes the first ``T_exec``, then re-plans.  The dataset must contain
matching state-action chunks of these dimensions.

References
----------
    - FinFlowRL (arXiv 2509.17964)
    - MeanFlow: Training Straight Normalizing Flows using Jacobian
      Regularization (Geng et al. 2025)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ======================================================================
# Dataset
# ======================================================================

class ExpertDataset(Dataset):
    """Dataset of expert demonstrations for flow matching pre-training.

    Each sample contains a window of ``T_obs`` market observations and a
    corresponding chunk of ``T_pred`` expert actions.  The model learns to
    generate the action chunk conditioned on the observation window.

    Parameters
    ----------
    states : np.ndarray
        Market observations, shape ``(N, T_obs, state_dim)``.
    actions : np.ndarray
        Expert actions, shape ``(N, T_pred, action_dim)``.
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray) -> None:
        if len(states) != len(actions):
            raise ValueError(
                f"States and actions must have the same number of samples: "
                f"got {len(states)} states and {len(actions)} actions."
            )
        if states.ndim != 3:
            raise ValueError(
                f"States must be 3-D (N, T_obs, state_dim), got shape {states.shape}."
            )
        if actions.ndim != 3:
            raise ValueError(
                f"Actions must be 3-D (N, T_pred, action_dim), got shape {actions.shape}."
            )

        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (state_window, action_chunk) for sample *idx*."""
        return self.states[idx], self.actions[idx]


# ======================================================================
# Learning rate schedule
# ======================================================================

class _CosineWarmupScheduler:
    """Linear warmup + cosine decay learning rate scheduler.

    After *warmup_steps* the LR decays from the peak value to
    *min_lr* following a cosine curve.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Number of steps with linear warmup.
    total_steps : int
        Total number of training steps (warmup + decay).
    min_lr_ratio : float
        Fraction of peak LR to decay to.  Default ``0.01``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)
        self.min_lr_ratio = min_lr_ratio
        self._current_step: int = 0

        # Store base LRs
        self._base_lrs: List[float] = [
            pg["lr"] for pg in self.optimizer.param_groups
        ]

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._current_step += 1
        step = self._current_step

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + np.cos(np.pi * progress)
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self) -> List[float]:
        """Return current learning rates for each param group."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Pre-trainer
# ======================================================================

class MeanFlowPretrainer:
    """Stage 1: Pre-train MeanFlow policy on expert demonstrations.

    Learns a flow-matching model that maps noise to expert actions,
    conditioned on market state, using the MeanFlow consistency loss
    with action chunking.

    Parameters
    ----------
    model : nn.Module
        A :class:`MeanFlowPolicy` instance to train.
    learning_rate : float
        Peak learning rate (after warmup).  Default ``3e-4``.
    weight_decay : float
        L2 regularisation coefficient.  Default ``1e-5``.
    batch_size : int
        Training batch size.  Default ``256``.
    num_epochs : int
        Maximum number of training epochs.  Default ``100``.
    warmup_steps : int
        Number of linear warmup steps.  Default ``1000``.
    checkpoint_dir : str
        Directory to save checkpoints.  Default ``"checkpoints/pretrain"``.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  Default ``"auto"``.
    grad_clip : float
        Maximum gradient norm for clipping.  Set to ``0.0`` to disable.
        Default ``1.0``.
    log_interval : int
        Log training metrics every N steps.  Default ``100``.
    eval_interval : int
        Run validation every N steps.  Set to ``0`` to disable.
        Default ``500``.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_epochs: int = 100,
        warmup_steps: int = 1000,
        checkpoint_dir: str = "checkpoints/pretrain",
        device: str = "auto",
        grad_clip: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 500,
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Training state
        self._global_step: int = 0
        self._best_val_loss: float = float("inf")
        self._history: List[Dict[str, float]] = []

        # Build optimizer (set up after dataset is known for total steps)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Placeholder scheduler (re-created in train())
        self._scheduler: Optional[_CosineWarmupScheduler] = None

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MeanFlowPretrainer initialised on {self.device}. "
            f"Model params: {self._count_parameters():,}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Implements the full MeanFlow training loop:

        1. Sample z_0 ~ N(0, I)
        2. Encode expert actions to latent space
        3. Sample flow times t ~ U(0, 1), r ~ U(0, t)
        4. Build straight-line interpolant z_t = (1-t)*z_0 + t*a_latent
        5. Predict velocity u_θ(z_t, r, t, s)
        6. Compute MeanFlow consistency loss
        7. Backpropagate and update parameters

        Parameters
        ----------
        batch_states : torch.Tensor
            Market observations, shape ``(B, T_obs, state_dim)``.
        batch_actions : torch.Tensor
            Expert actions, shape ``(B, T_pred, action_dim)``.

        Returns
        -------
        dict
            Dictionary with training metrics: ``loss``, ``lr``,
            ``grad_norm``, and velocity diagnostics from the model.
        """
        self.model.train()

        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)

        # Forward pass through MeanFlow policy
        loss, info = self.model(batch_states, batch_actions)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            ).item()
        else:
            # Compute grad norm without clipping for logging
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            grad_norm = total_norm ** 0.5

        # Optimiser step
        self.optimizer.step()

        # LR schedule
        if self._scheduler is not None:
            self._scheduler.step()

        self._global_step += 1

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            **info,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataset: ExpertDataset,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Evaluate the model on a validation dataset.

        Parameters
        ----------
        dataset : ExpertDataset
            Validation dataset.
        max_batches : int
            Maximum number of batches to evaluate (for speed).
            Default ``50``.

        Returns
        -------
        dict
            Dictionary with ``val_loss`` and diagnostic metrics.
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        total_loss = 0.0
        n_batches = 0

        for states, actions in loader:
            if n_batches >= max_batches:
                break
            states = states.to(self.device)
            actions = actions.to(self.device)
            loss, info = self.model(states, actions)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {"val_loss": avg_loss}

    def train(
        self,
        dataset: ExpertDataset,
        val_dataset: Optional[ExpertDataset] = None,
    ) -> Dict[str, Any]:
        """Full training loop with logging, validation, and checkpointing.

        Parameters
        ----------
        dataset : ExpertDataset
            Training dataset of expert demonstrations.
        val_dataset : ExpertDataset or None
            Optional validation dataset for early stopping / monitoring.

        Returns
        -------
        dict
            Training history with keys ``history`` (list of step metrics),
            ``best_val_loss``, ``total_steps``, and ``best_checkpoint``.
        """
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # Estimate total steps for cosine schedule
        steps_per_epoch = len(train_loader)
        total_steps = self.num_epochs * steps_per_epoch

        self._scheduler = _CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
        )

        logger.info(
            f"Starting pre-training: {self.num_epochs} epochs, "
            f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
        )
        logger.info(f"Training samples: {len(dataset)}")

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_losses: List[float] = []

            for batch_states, batch_actions in train_loader:
                metrics = self.train_step(batch_states, batch_actions)
                epoch_losses.append(metrics["loss"])

                # Log
                if self._global_step % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self._global_step / max(elapsed, 1e-6)
                    logger.info(
                        f"[Step {self._global_step:>7d}] "
                        f"loss={metrics['loss']:.6f} "
                        f"lr={metrics['lr']:.2e} "
                        f"grad_norm={metrics['grad_norm']:.4f} "
                        f"({steps_per_sec:.0f} steps/s)"
                    )

                # Validation
                if (
                    self.eval_interval > 0
                    and self._global_step % self.eval_interval == 0
                    and val_dataset is not None
                ):
                    val_metrics = self.evaluate(val_dataset)
                    logger.info(
                        f"  [Validation] val_loss={val_metrics['val_loss']:.6f}"
                    )
                    metrics["val_loss"] = val_metrics["val_loss"]

                    # Save best checkpoint
                    if val_metrics["val_loss"] < self._best_val_loss:
                        self._best_val_loss = val_metrics["val_loss"]
                        best_path = str(
                            self.checkpoint_dir / "best_model.pt"
                        )
                        self.save_checkpoint(best_path)
                        logger.info(
                            f"  New best val_loss={self._best_val_loss:.6f}, "
                            f"saved to {best_path}"
                        )

                self._history.append(metrics)

            # Epoch summary
            avg_epoch_loss = float(np.mean(epoch_losses))
            elapsed = time.time() - start_time
            logger.info(
                f"[Epoch {epoch:>4d}/{self.num_epochs}] "
                f"avg_loss={avg_epoch_loss:.6f} "
                f"elapsed={elapsed:.1f}s"
            )

        # Save final checkpoint
        final_path = str(self.checkpoint_dir / "final_model.pt")
        self.save_checkpoint(final_path)
        logger.info(f"Training complete. Final checkpoint: {final_path}")

        return {
            "history": self._history,
            "best_val_loss": self._best_val_loss,
            "total_steps": self._global_step,
            "best_checkpoint": str(
                self.checkpoint_dir / "best_model.pt"
            ),
            "final_checkpoint": final_path,
            "training_time": time.time() - start_time,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state to a checkpoint file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "best_val_loss": self._best_val_loss,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint file.

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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._global_step = checkpoint.get("global_step", 0)
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(
            f"Checkpoint loaded from {path} "
            f"(step={self._global_step}, best_val_loss={self._best_val_loss:.6f})"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ======================================================================
# Main – smoke test with synthetic data
# ======================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from finflowrl.models import MeanFlowPolicy

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  MeanFlowPretrainer – Smoke Test")
    print("=" * 70)

    # --- Create a small model ---
    model = MeanFlowPolicy(
        state_dim=14,
        action_dim=2,
        noise_dim=4,
        T_obs=2,
        T_pred=4,
        T_exec=2,
        hidden_dim=32,
        num_layers=2,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Create synthetic expert data ---
    N = 500
    states = np.random.randn(N, 2, 14).astype(np.float32)
    actions = np.random.uniform(0.01, 2.0, size=(N, 4, 2)).astype(np.float32)

    train_dataset = ExpertDataset(states[:400], actions[:400])
    val_dataset = ExpertDataset(states[400:], actions[400:])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- Create pre-trainer ---
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        pretrainer = MeanFlowPretrainer(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=32,
            num_epochs=3,
            warmup_steps=50,
            checkpoint_dir=tmpdir,
            device="cpu",
            log_interval=10,
            eval_interval=30,
            grad_clip=1.0,
        )

        # --- Test train_step ---
        print("\n--- Test: Single train_step ---")
        s_batch, a_batch = train_dataset[0], train_dataset[1]
        s_batch = torch.stack([s_batch, a_batch])  # wrong, fix:
        s_batch = torch.stack([train_dataset[i][0] for i in range(4)])
        a_batch = torch.stack([train_dataset[i][1] for i in range(4)])
        metrics = pretrainer.train_step(s_batch, a_batch)
        print(f"  Step metrics: {metrics}")
        assert "loss" in metrics, "Should have loss"
        assert "lr" in metrics, "Should have lr"
        assert "grad_norm" in metrics, "Should have grad_norm"
        assert np.isfinite(metrics["loss"]), "Loss should be finite"
        print("  [PASS]")

        # --- Test evaluate ---
        print("\n--- Test: Evaluate ---")
        val_metrics = pretrainer.evaluate(val_dataset, max_batches=5)
        print(f"  Val metrics: {val_metrics}")
        assert "val_loss" in val_metrics, "Should have val_loss"
        assert np.isfinite(val_metrics["val_loss"]), "Val loss should be finite"
        print("  [PASS]")

        # --- Test full train loop ---
        print("\n--- Test: Full train loop (3 epochs) ---")
        # Reset model
        model = MeanFlowPolicy(
            state_dim=14, action_dim=2, noise_dim=4,
            T_obs=2, T_pred=4, T_exec=2, hidden_dim=32, num_layers=2,
        )
        pretrainer = MeanFlowPretrainer(
            model=model,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=3,
            warmup_steps=50,
            checkpoint_dir=tmpdir,
            device="cpu",
            log_interval=10,
            eval_interval=50,
        )
        result = pretrainer.train(train_dataset, val_dataset)
        print(f"\n  Total steps: {result['total_steps']}")
        print(f"  Best val loss: {result['best_val_loss']:.6f}")
        print(f"  Training time: {result['training_time']:.2f}s")
        assert result["total_steps"] > 0, "Should have trained"
        print("  [PASS]")

        # --- Test checkpoint save/load ---
        print("\n--- Test: Checkpoint save/load ---")
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        pretrainer.save_checkpoint(ckpt_path)
        assert os.path.isfile(ckpt_path), "Checkpoint should exist"

        # Create new pre-trainer and load
        model2 = MeanFlowPolicy(
            state_dim=14, action_dim=2, noise_dim=4,
            T_obs=2, T_pred=4, T_exec=2, hidden_dim=32, num_layers=2,
        )
        pretrainer2 = MeanFlowPretrainer(
            model=model2, checkpoint_dir=tmpdir, device="cpu",
        )
        pretrainer2.load_checkpoint(ckpt_path)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
            assert torch.allclose(p1, p2, atol=1e-6), (
                f"Parameter {n1} mismatch after load"
            )
        print("  [PASS]")

    print("\n" + "=" * 70)
    print("  pretrain.py OK – all tests passed.")
    print("=" * 70)
