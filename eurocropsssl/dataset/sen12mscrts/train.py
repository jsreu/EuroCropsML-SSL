import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Literal

import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset

from eurocropsssl.dataset.base import custom_collate_fn
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.pixelsdataset import SEN12MSCRTSPixelDataset
from eurocropsssl.dataset.task import Task

logger = logging.getLogger(__name__)


def _load_sen12mscrts(
    train_folder_list: list[Path],
    val_folder_list: list[Path],
    dataset_config: SEN12MSCRTSDatasetConfig,
    max_masked_ratio: float | None,
    masking_strategy: Literal["random", "fixed"],
    model_channels: list[str] | None,
) -> Task:
    task = Task(
        task_id="sen12mscrts",
        train_set=TransformDataset(
            SEN12MSCRTSPixelDataset(
                folder_list=train_folder_list,
                config=dataset_config,
                model_channels=model_channels,
            ),
            collate_fn=partial(
                custom_collate_fn,
                max_masked_ratio=max_masked_ratio,
                model_channels=model_channels,
                masking_strategy=masking_strategy,
            ),
        ),
        val_set=TransformDataset(
            SEN12MSCRTSPixelDataset(
                folder_list=val_folder_list,
                config=dataset_config,
                model_channels=model_channels,
            ),
            collate_fn=partial(
                custom_collate_fn,
                max_masked_ratio=max_masked_ratio,
                model_channels=model_channels,
                masking_strategy=masking_strategy,
            ),
        ),
        num_classes=0,
        loss_fn=nn.MSELoss(),
        metrics=[],
    )
    return task


def load_dataset(
    dataset_config: SEN12MSCRTSDatasetConfig,
    masking_strategy: Literal["random", "fixed"],
    max_masked_ratio: float | None = 0.7,
    model_channels: list[str] | None = None,
) -> Task:
    """Load Sen12MS data.

    Args:
        dataset_config: Dataset config for pre-processed data.
        masking_strategy: Which masking strategy to use. "random" splits the masked tokens randomly
            between the items in the batch. "fixed" is used for the implementation of the Presto
            channel encoding and will mask the same number of tokens in each batch item.
        max_masked_ratio: Maximum ratio of masked elements in data.
            Defaults to 0.7.
        model_channels: List of used data bands/channels (e.g. S1, S2, ERA5) in the network.
           Used for potential NDVI calculation and group masking.

    Returns:
        Task containg train and validation dataset for self-supervised learning.
    """

    pixels_dir: Path = dataset_config.raw_data_dir.parent.joinpath("PixelSEN12MSCRTS")

    train_pkls: Path = pixels_dir.joinpath("train_paths.pkl")
    val_pkls: Path = pixels_dir.joinpath("val_paths.pkl")
    if not train_pkls.exists():
        logger.info("Collecting SEN12MS-CR-TS training ROI paths.")
        train_folders: list[Path] = sorted(
            [f for f in pixels_dir.joinpath("train").iterdir() if f.is_dir()]
        )
        with open(train_pkls, "wb") as f:
            pickle.dump(train_folders, f)
    else:
        logger.info("Loading SEN12MS-CR-TS training ROI paths.")
        with open(train_pkls, "rb") as f:  # type: ignore[assignment]
            train_folders = pickle.load(f)

    if not val_pkls.exists():
        logger.info("Collecting SEN12MS-CR-TS validation ROI paths.")
        val_folders: list[Path] = sorted(
            [f for f in pixels_dir.joinpath("val").iterdir() if f.is_dir()]
        )
        with open(val_pkls, "wb") as f:
            pickle.dump(val_folders, f)
    else:
        logger.info("Loading SEN12MS-CR-TS validation ROI paths.")
        with open(val_pkls, "rb") as f:  # type: ignore[assignment]
            val_folders = pickle.load(f)

    task = _load_sen12mscrts(
        train_folders,
        val_folders,
        dataset_config,
        max_masked_ratio,
        masking_strategy,
        model_channels,
    )

    return task
