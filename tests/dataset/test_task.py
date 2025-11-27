import pytest
import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset

from eurocropsssl.dataset.task import Task
from tests.utils import SineDataset


def test_task_dl() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    batch_size = 8
    num_batches = -(-len(transform_dataset) // batch_size)

    train_dl = task.train_dl(batch_size=batch_size)
    train_batch = next(iter(train_dl))
    assert len(train_dl) == num_batches
    assert train_batch[0].data.size(0) == batch_size
    assert train_batch[1].data.size(0) == batch_size

    test_dl = task.train_dl(batch_size=batch_size)
    test_batch = next(iter(test_dl))
    assert len(test_dl) == num_batches
    assert test_batch[0].data.size(0) == batch_size
    assert test_batch[1].data.size(0) == batch_size

    with pytest.raises(ValueError):
        task.val_dl(batch_size=batch_size)


def test_task_dl_from_num_batches() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    num_batches = 5
    batch_size = len(transform_dataset) // num_batches

    train_dl = task.train_dl(num_batches=num_batches)
    train_batch = next(iter(train_dl))
    assert len(train_dl) == num_batches
    assert train_batch[0].data.size(0) == batch_size
    assert train_batch[1].data.size(0) == batch_size

    test_dl = task.train_dl(num_batches=num_batches)
    test_batch = next(iter(test_dl))
    assert len(test_dl) == num_batches
    assert test_batch[0].data.size(0) == batch_size
    assert test_batch[1].data.size(0) == batch_size

    with pytest.raises(ValueError):
        task.val_dl(num_batches=num_batches)


def test_task_dl_val_dl() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        val_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    batch_size = 8
    num_batches = -(-len(transform_dataset) // batch_size)

    val_dl = task.val_dl(batch_size=batch_size)
    val_samples, val_labels = next(iter(val_dl))
    assert len(val_dl) == num_batches
    assert val_samples.data.size(0) == batch_size
    assert val_labels.data.size(0) == batch_size


def test_task_dl_val_dl_from_num_batches() -> None:
    transform_dataset = TransformDataset(SineDataset(phase=0.5, size=100))
    task = Task(
        task_id="test",
        train_set=transform_dataset,
        test_set=transform_dataset,
        val_set=transform_dataset,
        num_classes=1,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    num_batches = 5
    batch_size = len(transform_dataset) // num_batches

    val_dl = task.val_dl(num_batches=num_batches)
    val_samples, val_labels = next(iter(val_dl))
    assert len(val_dl) == num_batches
    assert val_samples.data.size(0) == batch_size
    assert val_labels.data.size(0) == batch_size
