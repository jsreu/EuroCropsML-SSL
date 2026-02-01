import logging
from typing import Literal

from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.train import load_dataset
from eurocropsssl.dataset.task import Task

logger = logging.getLogger(__name__)


def load_task(
    dataset_config: SEN12MSCRTSDatasetConfig,
    max_masked_ratio: float | None,
    masking_strategy: Literal["random", "fixed"],
    model_channels: list[str] | None,
) -> Task:
    """Load data from the SEN12MS-CR-TS dataset for self-supervised learning."""

    return load_dataset(
        dataset_config=dataset_config,
        max_masked_ratio=max_masked_ratio,
        masking_strategy=masking_strategy,
        model_channels=model_channels,
    )
