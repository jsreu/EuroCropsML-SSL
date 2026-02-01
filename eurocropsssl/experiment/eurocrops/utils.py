import logging
from pathlib import Path
from typing import Literal

import torch.nn as nn
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
    EuroCropsSplit,
)
from eurocropsml.dataset.preprocess import get_class_ids_to_names

from eurocropsssl.dataset.eurocrops.train import load_dataset_split
from eurocropsssl.dataset.task import Task

logger = logging.getLogger(__name__)


def load_task(
    split_dir: Path,
    mode: Literal["pretraining", "finetuning"],
    split_config: EuroCropsSplit,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfig,
    self_supervised: bool,
    presto_input: bool,
    max_masked_ratio: float | None,
    masking_strategy: Literal["random", "fixed", None],
    model_channels: list[str] | None,
    pretrained: bool = True,
    ndvi_flag: bool = True,
    max_samples: int | str = "all",
) -> Task:
    """Load pretraining or finetuning task from the EuroCrops dataset."""

    split = dataset_config.split

    classes = (
        set(split_config.pretrain_classes[split])
        if mode == "pretraining"
        else set(split_config.finetune_classes[split])
    )
    loss_fn = nn.MSELoss() if self_supervised else nn.CrossEntropyLoss()

    if presto_input:
        padding_value: float = -999.0
    else:
        padding_value = 0.0

    return load_dataset_split(
        mode=mode,
        classes=classes,
        split_dir=split_dir,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
        loss_fn=loss_fn,
        self_supervised=self_supervised,
        presto_input=presto_input,
        max_masked_ratio=max_masked_ratio,
        masking_strategy=masking_strategy,
        model_channels=model_channels,
        max_samples=max_samples,
        pretrained=pretrained,
        ndvi_flag=ndvi_flag,
        padding_value=padding_value,
        class_ids_to_names=get_class_ids_to_names(preprocess_config.raw_data_dir),
    )
