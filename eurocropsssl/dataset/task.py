from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeVar

import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset
from torch.utils.data import DataLoader

from eurocropsssl.train.utils import TaskMetric

logger = logging.getLogger(__name__)


TaskIx = TypeVar("TaskIx")


@dataclass
class Task:
    """Class specifying a training task.

    Args:
        task_id: Task identifier.
        train_set: Task train set.
        loss_fn: Loss function appropriate for task.
        metrics: Additional metrics evaluated during testing/validation.
        num_classes: Number of classes per task
        test_set: Optional task test set.
        val_set: Optional task validation set.
    """

    task_id: str
    train_set: TransformDataset
    loss_fn: nn.Module
    metrics: Sequence[TaskMetric]
    num_classes: int
    test_set: TransformDataset | None = None
    val_set: TransformDataset | None = None

    DATALOADER_NUM_WORKERS: ClassVar[int | None] = 0

    def _build_dl(
        self,
        ds: TransformDataset,
        batch_size: int,
        prefetch_factor: int | None = 2,
        mode: Literal["train", "test"] = "train",
    ) -> DataLoader:
        num_workers = self.DATALOADER_NUM_WORKERS
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            drop_last=False,
            collate_fn=ds.collate_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=bool(num_workers),
        )

    def train_dl(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Build dataloader serving train data with given batch size or number of batches."""
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.train_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.train_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.train_set, batch_size, prefetch_factor, mode="train")

    def val_dl(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Build dataloader serving validation data with given batch size or number of batches.

        Raises ValueError if val_set is None.
        """
        if self.val_set is None:
            raise ValueError(f"No validation set given for task {self.task_id}.")
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.val_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.val_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.val_set, batch_size, prefetch_factor, mode="test")

    def test_dl(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Build dataloader serving test data with given batch size or number of batches.

        Raises ValueError if test_set is None.
        """
        if self.test_set is None:
            raise ValueError(f"No test set given for task {self.task_id}.")
        if batch_size is None:
            if num_batches is None:
                raise ValueError(
                    "Either the batch size or the number of batches must be specified."
                )
            if len(self.test_set) % num_batches != 0:
                logger.warning(
                    "The number of samples in the dataset is not divisible "
                    "by the requested number of batches."
                )
            batch_size = len(self.test_set) // num_batches
            if batch_size < 1:
                logger.warning(
                    "Not enough samples in the dataset. Using fewer batches with a "
                    "minimum batch size of one instead. "
                )
                batch_size = 1
        return self._build_dl(self.test_set, batch_size, prefetch_factor, mode="test")
