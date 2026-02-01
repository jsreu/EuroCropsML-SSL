import logging
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import pandas as pd
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.dataset.dataset import EuroCropsDataset
from eurocropsml.dataset.preprocess import find_clouds
from eurocropsml.dataset.utils import MMapStore
from mmap_ninja.ragged import RaggedMmap

logger = logging.getLogger(__name__)


def _format_band_name(band: str) -> str:
    if band.isdigit():
        # For numeric bands, remove leading zeros
        return f"B{band.lstrip('0')}"
    else:
        # For special bands like "8A"
        return f"B{band}"


class MMapStoreERA5(MMapStore):
    """Memory map metadata class for ERA5 processing.

    Args:
        file_paths: Iterable of file paths to memory map.
        era5_folder: Folder with ERA5 data.

    """

    def __init__(self, file_paths: Iterable[Path], era5_folder: Optional[Path] = None) -> None:
        self.era5_folder = era5_folder
        super().__init__(file_paths)

        if self.era5_folder and self.era5_folder.exists() and "era5" not in self.mmaps:
            if not (self.mmap_data_dir / "era5").is_dir():
                logger.info("Existing MMap does not yet contain ERA5 data. Will create it.")
                self._process_array_type("era5")
            self.mmaps["era5"] = RaggedMmap(
                out_dir=self.mmap_data_dir / "era5",
                copy_before_wrapper_fn=False,
            )
        else:
            logger.info(
                "No ERA5 directory found. Please download ERA5 data if you want to include it."
            )

    def _load_era5_npz(self, npz_file: Path) -> np.ndarray:
        era5_file = self.era5_folder / npz_file.name if self.era5_folder else None
        if era5_file and era5_file.exists():
            try:
                with np.load(era5_file, mmap_mode="r") as era5_data:
                    temperature = era5_data["temperature"]
                    precipitation = era5_data["precipitation"]
                    return cast(np.ndarray, np.stack([temperature, precipitation], axis=-1))
            except Exception as e:
                logger.error(f"Failed to load ERA5 for {npz_file.name}: {e}")
        else:
            logger.warning(f"{era5_file} missing. Skipping.")
        return np.empty((0, 2), dtype=np.float32)

    def _process_array_type(self, array_name: str) -> None:
        if array_name != "era5":
            super()._process_array_type(array_name)
            return

        RaggedMmap.from_generator(
            out_dir=self.mmap_data_dir / array_name,
            sample_generator=(
                self._load_era5_npz(Path(file_str))
                for file_str in self.mmap_store_metadata.sorted_used_files
            ),
            batch_size=1024,
            verbose=True,
        )


class EuroCropsDatasetERA5(EuroCropsDataset):
    """EuroCropsDataset for ERA5 processing."""

    @staticmethod
    def _format_dates(
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        s2_data_bands: list[str] | None,
        date_type: Literal["day", "month"],
        arrays_dict: dict[str, dict[str, np.ndarray]],
        padding_value: float = 0.0,
    ) -> dict[str, dict[str, np.ndarray]]:

        match date_type:
            case "day":
                for satellite, dates in arrays_dict["dates"].items():
                    arrays_dict["dates"][satellite] = (
                        pd.to_datetime(dates).day_of_year.to_numpy() - 1
                    )
            case "month":
                for satellite, data in arrays_dict["data"].items():
                    dates = arrays_dict["dates"][satellite]
                    dates_month = pd.to_datetime(dates).month.to_numpy() - 1
                    unique, unique_indices, unique_counts = np.unique(
                        dates_month, return_index=True, return_counts=True
                    )

                    month_data = np.full((12, *data.shape[1:]), padding_value, dtype=np.float32)
                    if "era5" in arrays_dict:
                        month_era5 = np.full(
                            (12, *arrays_dict["era5"][satellite].shape[1:]),
                            padding_value,
                            dtype=np.float32,
                        )
                        era5_source = arrays_dict["era5"][satellite]
                        filled_months = np.zeros(12, dtype=bool)
                    else:
                        month_era5 = None

                    for month, count, idx in zip(unique, unique_counts, unique_indices):
                        if count == 1:
                            month_data[month] = data[idx]
                            if month_era5 is not None:
                                month_era5[month] = era5_source[idx]
                                filled_months[month] = True
                        elif satellite == "S2":
                            s2_data_bands = cast(list[str], s2_data_bands)
                            try:
                                cloud_probs = np.apply_along_axis(
                                    partial(
                                        find_clouds,
                                        band4_idx=s2_data_bands.index("04"),
                                        preprocess_config=preprocess_config,
                                    ),
                                    1,
                                    data[idx : idx + count],
                                )
                            except ValueError as err:
                                raise ValueError(
                                    "Band 4 cannot be excluded if date_type is 'months'"
                                ) from err
                            best_idx = idx + np.argmin(cloud_probs)
                            month_data[month] = data[best_idx]
                            if month_era5 is not None:
                                month_era5[month] = era5_source[best_idx]
                                filled_months[month] = True
                        else:
                            month_data[month] = data[idx]
                            if month_era5 is not None:
                                month_era5[month] = era5_source[idx]
                                filled_months[month] = True

                    # Fill missing era5 values with monthly mean
                    if month_era5 is not None:
                        year: int = preprocess_config.year
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            days: int = 366
                        else:
                            days = 365
                        full_era5_dates = pd.date_range(
                            f"{str(year)}-01-01", periods=days, freq="D"
                        )
                        full_era5_months = full_era5_dates.month.to_numpy() - 1

                        for month in range(12):
                            if np.all(month_era5[month] == 0):
                                month_indices = np.where(full_era5_months == month)[0]
                                if len(month_indices) > 0:
                                    month_mean = era5_source[month_indices].mean(axis=0)
                                    month_era5[month] = month_mean

                    arrays_dict["data"][satellite] = month_data
                    arrays_dict["dates"][satellite] = np.arange(12)
                    if month_era5 is not None:
                        arrays_dict["era5"][satellite] = month_era5

        return arrays_dict
