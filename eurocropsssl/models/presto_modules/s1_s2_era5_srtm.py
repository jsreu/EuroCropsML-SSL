"""Taken from original Presto code.

https://github.com/nasaharvest/presto/blob/main/presto/dataops/pipelines/s1_s2_era5_srtm.py
"""

import warnings
from collections import OrderedDict
from functools import partial
from typing import List, cast
from typing import OrderedDict as OrderedDictType

import numpy as np

from .ee_pipeline import EEPipeline

try:
    import torch

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

"""
For easier normalization of the band values (instead of needing to recompute
the normalization dict with the addition of new data), we provide maximum
values for each band
"""
S1_BANDS = ["VV", "VH"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)
ERA5_BANDS = ["2m_temperature", "total_precipitation"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
SRTM_BANDS = ["elevation", "slope"]
# visually gauged 90th percentile from
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]

DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS

DYNAMIC_BANDS_SHIFT = S1_SHIFT_VALUES + S2_SHIFT_VALUES + ERA5_SHIFT_VALUES
DYNAMIC_BANDS_DIV = S1_DIV_VALUES + S2_DIV_VALUES + ERA5_DIV_VALUES

STATIC_BANDS_SHIFT = SRTM_SHIFT_VALUES
STATIC_BANDS_DIV = SRTM_DIV_VALUES

# These bands are what is created by the Engineer. If the engineer changes, the bands
# here will need to change (and vice versa)
REMOVED_BANDS = ["B1", "B10"]
RAW_BANDS = DYNAMIC_BANDS + STATIC_BANDS

BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + STATIC_BANDS + ["NDVI"]
# NDVI is between 0 and 1
ADD_BY = (
    [DYNAMIC_BANDS_SHIFT[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_SHIFT
    + [0.0]
)
DIVIDE_BY = (
    [DYNAMIC_BANDS_DIV[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + STATIC_BANDS_DIV
    + [1.0]
)

NUM_TIMESTEPS = 12
NUM_ORG_BANDS = len(BANDS)
TIMESTEPS_IDX = list(range(NUM_TIMESTEPS))

NORMED_BANDS = [x for x in BANDS if x != "B9"]
NUM_BANDS = len(NORMED_BANDS)
BANDS_IDX = list(range(NUM_BANDS))
BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [NORMED_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [NORMED_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [NORMED_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [NORMED_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [NORMED_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [NORMED_BANDS.index(b) for b in ["B11", "B12"]],  # Include B10?
        "ERA5": [NORMED_BANDS.index(b) for b in ERA5_BANDS],
        "SRTM": [NORMED_BANDS.index(b) for b in SRTM_BANDS],
        "NDVI": [NORMED_BANDS.index("NDVI")],
    }
)


class S1_S2_ERA5_SRTM(EEPipeline):
    """Pipeline for processing and combining the multi data sources.

    Sentinel-1, Sentinel-2, ERA5 climate data, SRTM elevation data and NDVI calculation.

    """

    item_shape = (NUM_TIMESTEPS, NUM_BANDS)

    @staticmethod
    def calculate_ndvi(
        input_array: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        r"""Calculate NDVI from S2 bands B04 and B08.

        Given an input array of shape [timestep, bands] or [batches, timesteps, shapes]
        where bands == len(bands), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)

        """
        band_1, band_2 = "B8", "B4"

        num_dims = len(input_array.shape)
        if num_dims == 2:
            band_1_np = input_array[:, NORMED_BANDS.index(band_1)]
            band_2_np = input_array[:, NORMED_BANDS.index(band_2)]
        elif num_dims == 3:
            band_1_np = input_array[:, :, NORMED_BANDS.index(band_1)]
            band_2_np = input_array[:, :, NORMED_BANDS.index(band_2)]
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            if isinstance(band_1_np, np.ndarray):
                return np.where(
                    (band_1_np + band_2_np) > 0,
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    0,
                )
            else:
                return torch.where(
                    (band_1_np + band_2_np) > 0,  # type: ignore[arg-type]
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),  # type: ignore[arg-type]
                    0,
                )

    @classmethod
    def normalize(
        cls,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Normalize satellite imagery, remove B9 band, apply scaling factors, add NDVI."""
        # remove the b9 band
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        if isinstance(x, np.ndarray):
            x = ((x + ADD_BY) / DIVIDE_BY).astype(np.float32)
            cast_ndvi = partial(cast, np.ndarray)
        else:
            x = cast(torch.Tensor, (x + torch.tensor(ADD_BY)) / torch.tensor(DIVIDE_BY))
            cast_ndvi = partial(cast, torch.Tensor)

        if len(x.shape) == 2:
            x = x[:, keep_indices]
            ndvi = cast_ndvi(cls.calculate_ndvi(x))
            x[:, NORMED_BANDS.index("NDVI")] = ndvi
        else:
            x = x[:, :, keep_indices]
            ndvi = cast_ndvi(cls.calculate_ndvi(x))
            x[:, :, NORMED_BANDS.index("NDVI")] = ndvi
        return x
