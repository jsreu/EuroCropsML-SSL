import logging
from pathlib import Path

import numpy as np
import xarray as xr

from eurocropsssl.settings import EPSILON

logger = logging.getLogger(__name__)

MAX_TEMP = 324
MIN_TEMP = 239
MAX_PRECIP = 0.03


def load_era5_data(date: str, base_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Load ERA5 analysis and forecast data for the given date.

    Args:
        date: Date string in "YYYY-MM-DD" format.
        base_path: Path to the directory containing ERA5 .grb files.

    Returns:
        A tuple of xarray Datasets: (analysis dataset, forecast dataset).
    """

    year, month, day = date.split("-")

    filename = base_path.joinpath(f"era5_{year}_{month}_{day}.grb")

    if not filename.exists():
        raise FileNotFoundError(
            f"Data file missing: {filename}. Please first download the ERA5 "
            "data and make sure all files were downloaded correctly."
        )
    with (
        xr.open_dataset(
            filename,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"dataType": "an"}, "indexpath": ""},
            decode_timedelta=True,
        ) as ds_an,
        xr.open_dataset(
            filename,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"dataType": "fc"}, "indexpath": ""},
            decode_timedelta=True,
        ) as ds_fc,
    ):
        return ds_an, ds_fc


def extract_values(
    ds_an: xr.Dataset,
    ds_fc: xr.Dataset,
    bbox: tuple[float, float, float, float],
    normalize: bool = False,
) -> list[float]:
    """Extract ERA5 variables for the given bounding box.

    Args:
        ds_an: Analysis dataset.
        ds_fc: Forecast dataset.
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        normalize: Whether to normalizxe the values.

    Returns:
        A list of normalized variable values extracted from both datasets.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Extract analysis variables
    variables_an = {}
    var = sorted(ds_an.data_vars)  # make sure they always have the same order
    for var_name in var:
        variables_an[var_name] = (
            ds_an[var_name]
            .interp(latitude=center_lat, longitude=center_lon)  # interpolate missing values
            .mean(dim="time")
            .values.item()
        )

        if var_name == "t2m" and normalize:  # temperature normalization
            variables_an[var_name] = np.clip(variables_an[var_name], MIN_TEMP, MAX_TEMP)
            variables_an[var_name] = (variables_an[var_name] - MIN_TEMP) / (MAX_TEMP - MIN_TEMP) * (
                1 - EPSILON
            ) + EPSILON

    variables_fc = {}
    var = sorted(ds_fc.data_vars)  # make sure they always have the same order
    for var_name in var:
        variables_fc[var_name] = np.nansum(
            ds_fc[var_name]
            .interp(latitude=center_lat, longitude=center_lon)  # interpolate missing values
            .values
        )

        if var_name == "tp" and normalize:  # precipitation normalization
            variables_fc[var_name] = np.clip(variables_fc[var_name], 0, MAX_PRECIP)
            variables_fc[var_name] = (variables_fc[var_name] / MAX_PRECIP) * (1 - EPSILON) + EPSILON

    return list(variables_an.values()) + list(variables_fc.values())


def get_era5_values(
    date: str,
    bbox: tuple[float, float, float, float],
    base_path: Path,
    normalize: bool = False,
) -> list[float]:
    """Extracts ERA5 temperature and precipitation values for a single location.

    Args:
        date: Date in "YYYY-MM-DD".
        bbox: Geographic bounds (min_lon, min_lat, max_lon, max_lat).
        base_path: Directory containing ERA5 files.
        normalize: Whether to normalizxe the values.

    Returns:
        Extracted ERA5 values for the bbox.
    """

    ds_an, ds_fc = load_era5_data(date, base_path)
    result = extract_values(ds_an, ds_fc, bbox, normalize)

    ds_an.close()
    ds_fc.close()

    return result
