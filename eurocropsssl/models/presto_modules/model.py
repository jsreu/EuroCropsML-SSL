"""Taken from original Presto code.

https://github.com/nasaharvest/presto/blob/main/presto/model.py
"""

import torch
from torch import nn


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model combining an encoder and decoder.

    A standard encoder-decoder architecture for sequence transformation tasks.

    """

    encoder: nn.Module
    decoder: nn.Module


class FinetuningHead(nn.Module):
    """Classification head for fine-tuning tasks.

    Simple projection layer with optional sigmoid activation for binary classification.

    """

    def __init__(self, hidden_size: int, num_outputs: int, regression: bool):
        """Initialize fine-tuning head.

        Args:
            hidden_size: Dimension of input features
            num_outputs: Number of output dimensions
            regression: If True, outputs raw values; if False, applies activation for classification
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.regression = regression
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fine-tuning head.

        Args:
            x: Input tensor of embeddings

        Returns:
            Prediction tensor with sigmoid activation applied for binary classification
        """
        x = self.linear(x)
        if (not self.regression) & (self.num_outputs == 1):
            x = torch.sigmoid(x)
        return x


class FineTuningModel(nn.Module):
    """Base class for fine-tuning models with a shared encoder architecture.

    Abstract base class that defines the encoder component interface.

    """

    encoder: nn.Module
