import calendar
import logging
import multiprocessing as mp_orig
import time
from functools import partial
from pathlib import Path
from typing import Literal, cast

import cdsapi
from tqdm import tqdm

from eurocropsssl.dataset.era5.config import ERA5DatasetConfig

logger = logging.getLogger(__name__)


def download_with_retry(
    output_path: Path,
    bbox: tuple[float, float, float, float],
    climate_variables: list[Literal["2m_temperature", "total_precipitation"]],
    dates: tuple[int, int, int],
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
) -> None:
    """Download ERA5 data for a specific day with configurable retries.

    Args:
        output_path: The directory where the downloaded file will be saved.
        dates: A tuple containing the year, month, and day (YYYY, MM, DD).
        bbox: Geographic bounding box in the format [N, W, S, E].
        climate_variables: List of ERA5 climate variables to download.
        max_attempts: Maximum number of download attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplicative factor for exponential delay between retries.
    """
    year, month, day = dates

    options = {
        "product_type": "reanalysis",
        "variable": climate_variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": f"{day:02d}",
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": bbox,
        "grid": [0.25, 0.25],
        "format": "grib",
    }

    filename = output_path.joinpath(f"era5_{year}_{month:02d}_{day:02d}.grb")
    if Path(filename).exists():
        logger.info(f"{year}-{month:02d}-{day:02d}: Already exists")
        return

    delay = initial_delay
    last_exception = None

    for attempt in range(max_attempts):
        try:
            c = cdsapi.Client()
            c.retrieve("reanalysis-era5-single-levels", options, filename)
            return

        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:  # Don't log "retrying" on the last attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for "
                    f"{year}-{month}-{day}: {str(e)}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    f"All {max_attempts} attempts failed for {year}-{month}-{day}. "
                    f"Last error: {str(e)}. Please check your CDSAPI credentials "
                    f"and ensure you have enabled the Licence to use Copernicus Products "
                    f"in your profile: https://cds.climate.copernicus.eu/profile"
                )

    if last_exception:
        raise last_exception


def download_era5(config: ERA5DatasetConfig) -> None:
    """Download ERA5 data for a specific dataset and year with parallel processing.

    Args:
        config: Configuration settings for ERA5.
    """
    output_path = config.raw_data_dir
    base_dataset = config.base_dataset
    logger.info(f"Downloading ERA5 data for dataset {base_dataset} for the year {config.year}...")

    output_path.mkdir(parents=True, exist_ok=True)
    bbox = config.bbox
    variables = config.climate_variables

    if (
        base_dataset == "sen12mscrts"
        or base_dataset == "eurocrops_estonia"
        or base_dataset == "eurocrops_portugal"
        or base_dataset == "eurocrops_latvia"
    ):
        dates = [
            (config.year, month, day)
            for month in range(1, 13)
            for day in range(1, calendar.monthrange(cast(int, config.year), month)[1] + 1)
        ]

        func = partial(download_with_retry, output_path, bbox, variables)
        max_workers = min(mp_orig.cpu_count(), max(1, config.workers))

        with mp_orig.Pool(processes=max_workers) as p:
            te = tqdm(
                total=len(dates),
                desc="Downloading ERA5...",
            )
            te.refresh()

            for _ in p.imap_unordered(func, dates):
                te.update(n=1)
            te.close()
    else:
        raise NotImplementedError(f"ERA5 data collection is not implemented for {base_dataset}.")
