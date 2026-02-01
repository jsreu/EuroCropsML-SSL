import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationInfo, field_validator

from eurocropsssl.settings import Settings

logger = logging.getLogger(__name__)


class SEN12MSCRTSDatasetConfig(BaseModel):
    """Configuration for downloading, preprocessing, and using SEN12MS-CR-TS dataset.

    Args:
        raw_data_dir: Directory where the raw SEN12MS-CR-TS data is stored.
        era5_dir: Directory where global ERA5 data is stored.
        era5_normalize: Whether to scale ERA5 data to (0, 1)
        workers: Max workers used for parallel pre-processing.
        original_patch_size: Original height/width of SEN12MS-CR-TS patches.
        subset_ratio: Ratio used for sampling a subset of the full SEN12MS-CR-TS pixel dataset.
            The sampling is performed on patch-level within the train and validation set,
            respectively. If 1.0, the full dataset is used. Defaults to 0.25.
        splits: Which splits to get data for: train, validation, test or all split.
            So far, the pixel-wise split only supports `all`.
        use_sentinel1_data: Flag for obtaining and using Sentinel-1 for the datasets.
        region: Which region(s) to obtain data for.
        cloud_masks: Type of cloud mask detector to run on optical data
        sample_type: Whether to sample generic (returning full unfiltered data) or cloudfree.

    """

    raw_data_dir: Path
    era5_dir: Path
    era5_normalize: bool

    workers: int
    original_patch_size: int = 256
    subset_ratio: float = 0.25

    splits: Literal["all"] = "all"  # TODO: so far, the pixel-wise split only supports `all`
    use_sentinel1_data: bool = True

    region: Literal["all", "africa", "america", "asiaEast", "asiaWest", "europa"] = "europa"
    cloud_masks: Literal["cloud_cloudshadow_mask", "s2cloudless_map", "s2cloudless_mask"] = (
        "s2cloudless_mask"
    )
    sample_type: Literal["generic", "cloudy_cloudfree"] = "generic"

    @field_validator("raw_data_dir", "era5_dir")
    @classmethod
    def relative_path(cls, v: Path, info: ValidationInfo) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            if info.field_name == "era5_dir":
                v = Settings().era5_data_dir.joinpath(v)
            else:
                v = Settings().sen12ms_data_dir.joinpath(v)
        return v
