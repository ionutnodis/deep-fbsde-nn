"""
Checkpointing Utilities
=======================

Model saving, loading, and experiment tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save a training checkpoint.

    Args:
        path: Save path
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch/iteration
        loss: Current loss
        metadata: Additional metadata dict
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to

    Returns:
        Checkpoint dict with epoch, loss, metadata
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metadata": checkpoint.get("metadata", {}),
        "timestamp": checkpoint.get("timestamp"),
    }


class ExperimentLogger:
    """
    Simple experiment logger for tracking runs.

    Saves training history and metadata to JSON.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.json"

        self.data = {
            "name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "history": [],
            "results": {},
        }

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.data["config"] = config
        self._save()

    def log_step(self, iteration: int, loss: float, **kwargs):
        """Log a training step."""
        entry = {"iteration": iteration, "loss": loss, **kwargs}
        self.data["history"].append(entry)

    def log_result(self, key: str, value: Any):
        """Log a final result."""
        self.data["results"][key] = value
        self._save()

    def finish(self):
        """Finalize and save the log."""
        self.data["end_time"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2, default=str)


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to latest checkpoint, or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)
