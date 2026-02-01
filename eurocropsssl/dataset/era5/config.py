import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from eurocropsssl.settings import Settings

logger = logging.getLogger(__name__)


class ERA5DatasetConfig(BaseModel):
    """Configuration for downloading and preprocessing ERA5 dataset.

    Args:
        raw_data_dir: Directory where the raw ERA5 data is stored.
        base_dataset: Name of the dataset to collect ERA5 data for.
        workers: Max workers used for parallel pre-processing.
        chunk_size: Chunk size used for multiprocessing of EuroCrops ERA5 data.
        year: Year to collect ERA5 data for.
        bbox: Geographic bounding box in the format [W, S, E, N].
            Defaults to bbox for the whole world.
        climate_variables: List of ERA5 climate variables to download.
        normalize: Whether to scale the climate variables to (0, 1).
    """

    raw_data_dir: Path
    base_dataset: Literal[
        "sen12mscrts", "eurocrops_estonia", "eurocrops_portugal", "eurocrops_latvia"
    ] = "sen12mscrts"

    workers: int
    chunk_size: int = 10000
    year: int | None = None
    bbox: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
    climate_variables: list[Literal["2m_temperature", "total_precipitation"]] = [
        "2m_temperature",
        "total_precipitation",
    ]
    normalize: bool = False

    @field_validator("raw_data_dir")
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        v = Settings().era5_data_dir.joinpath(v)
        return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.post_init()

    def post_init(self) -> None:
        """Post initialization."""
        if self.base_dataset == "sen12mscrts":
            self.year = 2018
            logger.info("Setting bbox for SEN12MS-CR-TS...")
            # TODO: This is currently only for Europe
            bbox = (-11.0, 39.0, 30.0, 55.0)  # bbox for SEN12MS-CR-TS dataset
            self.year = 2018
        elif self.base_dataset == "eurocrops_estonia":
            self.year = 2021
            logger.info("Setting bbox for EuroCrops Estonia...")
            bbox = (20.37, 57.52, 28.2, 60)  # bbox for Estonia
        elif self.base_dataset == "eurocrops_portugal":
            self.year = 2021
            logger.info("Setting bbox for EuroCrops Portugal...")
            bbox = (-9.559, 36.961, -6.189, 42.154)  # bbox for Portugal
        elif self.base_dataset == "eurocrops_latvia":
            self.year = 2021
            logger.info("Setting bbox for EuroCrops Latvia...")
            bbox = (20.968, 55.674, 28.241, 58.085)  # bbox for Latvia
        logger.info("Re-ordering bounding box coordinates to [N, W, S, E] for ERA5...")
        self.bbox = (bbox[3], bbox[0], bbox[1], bbox[2])
