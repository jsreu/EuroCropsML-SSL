import math
from typing import cast

import torch
import torch.nn as nn
from eurocropsml.dataset.base import DataItem, LabelledData
from torch.utils.data import Dataset

from eurocropsssl.models.base import ModelBuilder


class SineDataset(Dataset[LabelledData]):
    """Toy dataset modeling a sine curve.

    Args:
        phase: Phase of sine curve
        size: Number of data points to construct.
    """

    def __init__(self, phase: float, size: int = 100):
        self.data = math.pi * torch.rand((size, 1))
        self.targets = torch.sin(self.data + phase)

    def __getitem__(self, ix: int) -> LabelledData:
        return LabelledData(DataItem(self.data[ix]), self.targets[ix])

    def __len__(self) -> int:
        return self.data.size(0)


class DataItemToTensor(nn.Module):
    """Module for getting a specified attribute from a DataItem.

    Args:
        model_input_att: Name of attribute to get from DataItem
        metadata: True if attribute is a metadata attribute
    """

    def __init__(self, model_input_att: str, metadata: bool = False):
        super().__init__()
        self.model_input_att = model_input_att
        self.metadata = metadata

    def forward(self, ipt: DataItem) -> torch.Tensor:
        """Forward pass."""
        if self.metadata:
            ipt = getattr(ipt, self.model_input_att)
        return cast(torch.Tensor, getattr(ipt, self.model_input_att))


class DenseNNBuilder(ModelBuilder):
    """Builder for two-layer dense neural network.

    Args:
        hidden_size: Size of hidden layer
        out_size: Size of output layer
    """

    def __init__(self, in_size: int, hidden_size: int = 32, out_size: int = 1):
        self._in_size = in_size
        self._hidden_size = hidden_size
        self._out_size = out_size

    def build_backbone(self) -> nn.Sequential:
        """Backbone to pre-train."""
        return nn.Sequential(
            DataItemToTensor("data"),
            nn.Linear(self._in_size, self._hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self._hidden_size, self._hidden_size),
            nn.ReLU(inplace=True),
        )

    def build_classification_head(self, num_classes: int) -> nn.Linear:
        """Prediction head."""
        return nn.Linear(self._hidden_size, num_classes)
