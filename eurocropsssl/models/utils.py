import torch
from torch import nn


def get_device(module: nn.Module) -> torch.device:
    """Get device module is loaded on."""
    return next(module.parameters()).device
