from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, cast

import torch
from eurocropsml.dataset.base import DataItem
from torch import nn

from eurocropsssl.utils import BaseConfig


class ModelConfig(BaseConfig):
    """Base class for model configs."""

    model_builder: str

    class Config:
        """Configuration base class.

        Defines protected namespaces that should not be modified
        during configuration processing.
        """

        protected_namespaces = ()


class Model(nn.Module):
    """Generic model architecture for pretraining and finetuning.

    Args:
        backbone: Model backbone to pretrain/finetune.
        head: Task-specific classification head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.to(device)
        self.device = device

    def forward(self, ipt: DataItem) -> torch.Tensor:
        out = self.backbone(ipt.data, **ipt.meta_data)
        return cast(torch.Tensor, self.head(out))

    def save(self, checkpoint: Path) -> None:
        """Save model to given checkpoint directory."""
        if not checkpoint.is_dir():
            raise ValueError("Model checkpoint must be a directory.")

        torch.save(
            self.backbone.state_dict(),
            checkpoint.joinpath("backbone.pt"),
        )
        torch.save(
            self.head.state_dict(),
            checkpoint.joinpath("head.pt"),
        )

    def load(self, checkpoint: Path, load_head: bool = True) -> None:
        """Load weights from checkpoint.

        Args:
            checkpoint: Path to directory containing model weights.
            load_head: Whether to also load weights for the model head.
        """

        backbone_state = torch.load(checkpoint.joinpath("backbone.pt"), map_location=self.device)
        self.backbone.load_state_dict(backbone_state, strict=False)
        if load_head and (head_checkpoint := checkpoint.joinpath("head.pt")).is_file():
            head_state = torch.load(head_checkpoint, map_location=self.device)
            self.head.load_state_dict(head_state)


T = TypeVar("T", bound=ModelConfig)


class ModelBuilder(ABC, Generic[T]):
    """Abstract base class for building models."""

    @abstractmethod
    def __init__(self, config: T):
        self.config = config

    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Build model backbone."""

    @abstractmethod
    def build_classification_head(self, num_classes: int) -> nn.Module:
        """Build classification head."""

    def build_self_supervised_head(self) -> nn.Module:
        """Build self-supervised head."""
        raise NotImplementedError

    def build_self_supervised_model(self, device: torch.device) -> Model:
        """Build model suitable for self-supervised training."""
        backbone = self.build_backbone()
        head = self.build_self_supervised_head()
        return Model(backbone=backbone, head=head, device=device)

    def build_classification_model(self, num_classes: int, device: torch.device) -> Model:
        """Build model suitable for classification with `num_classes` classes."""

        backbone = self.build_backbone()
        head = self.build_classification_head(num_classes)
        return Model(backbone=backbone, head=head, device=device)

    def reset_head(self, model: Model, num_classes: int) -> None:
        """Reset model classification head."""
        head = self.build_classification_head(num_classes)
        head.to(model.device)
        model.head = head
