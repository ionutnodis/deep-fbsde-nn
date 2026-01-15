"""
Device Management
=================

Utilities for managing compute devices (CUDA, MPS, CPU).
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(preference)


def setup_device(device: torch.device) -> dict:
    """
    Setup device-specific optimizations.

    Args:
        device: The device to configure

    Returns:
        dict with device capabilities
    """
    capabilities = {
        "device": device,
        "amp_available": False,
        "compile_available": False,
        "dtype": torch.float32,
    }

    if device.type == "cuda":
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        capabilities["amp_available"] = True
        capabilities["compile_available"] = int(torch.__version__.split(".")[0]) >= 2

        # Get GPU info
        capabilities["gpu_name"] = torch.cuda.get_device_name(0)
        capabilities["gpu_memory"] = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )

    elif device.type == "mps":
        # MPS (Apple Silicon)
        capabilities["amp_available"] = False  # AMP not fully supported on MPS
        capabilities["compile_available"] = False

    else:
        # CPU
        torch.set_num_threads(8)
        capabilities["amp_available"] = False
        capabilities["compile_available"] = int(torch.__version__.split(".")[0]) >= 2

    return capabilities


def print_device_info(device: torch.device):
    """Print device information."""
    caps = setup_device(device)

    print(f"Device: {device}")
    if "gpu_name" in caps:
        print(f"  GPU: {caps['gpu_name']}")
        print(f"  Memory: {caps['gpu_memory']:.1f} GB")
    print(f"  AMP available: {caps['amp_available']}")
    print(f"  torch.compile available: {caps['compile_available']}")
