"""Taken from original Presto code.

https://github.com/nasaharvest/presto/blob/main/presto/utils.py
"""

import logging
from typing import List, Optional, Tuple, cast

import torch

from eurocropsssl.models.presto_modules.s1_s2_era5_srtm import (
    BANDS,
    ERA5_BANDS,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    SRTM_BANDS,
)

logger = logging.getLogger("__main__")

dynamic_world_class_amount = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED: int = 42


def construct_single_presto_input(
    s1: Optional[torch.Tensor] = None,
    s1_bands: Optional[List[str]] = None,
    s2: Optional[torch.Tensor] = None,
    s2_bands: Optional[List[str]] = None,
    era5: Optional[torch.Tensor] = None,
    era5_bands: Optional[List[str]] = None,
    srtm: Optional[torch.Tensor] = None,
    srtm_bands: Optional[List[str]] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    normalize: bool = True,
    padding_value: float = -999.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct Presto input.

    Inputs are paired into a tensor input <X>, and a list <X>_bands, which describes <X>.

    <X> should have shape (num_timesteps, len(<X>_bands)), with the following bands possible for
    each input:

    s1: ["VV", "VH"]
    s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    era5: ["2m_temperature", "total_precipitation"]
        "2m_temperature": Temperature of air at 2m above the surface of land,
            sea or in-land waters in Kelvin (K)
        "total_precipitation": Accumulated liquid and frozen water, including rain and snow,
            that falls to the Earth's surface. Measured in metres (m)
    srtm: ["elevation", "slope"]

    dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
        of each timestep for that pixel
    """
    num_timesteps_list = [x.shape[0] for x in [s1, s2, era5, srtm] if x is not None]
    if dynamic_world is not None:
        num_timesteps_list.append(len(dynamic_world))

    assert len(num_timesteps_list) > 0
    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    num_timesteps = num_timesteps_list[0]
    mask, x = torch.ones(num_timesteps, len(BANDS)), torch.zeros(num_timesteps, len(BANDS))

    for band_group in [
        (s1, s1_bands, S1_BANDS),
        (s2, s2_bands, S2_BANDS),
        (era5, era5_bands, ERA5_BANDS),
        (srtm, srtm_bands, SRTM_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue
        kept_output_bands = [x for x in output_bands if x not in REMOVED_BANDS]
        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in kept_output_bands]
        kept_input_band_names = [val for val in input_bands if val in kept_output_bands]

        input_to_output_mapping = [BANDS.index(val) for val in kept_input_band_names]

        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        pad_mask = data == padding_value
        mask[:, input_to_output_mapping] = pad_mask.float()

    if dynamic_world is None:
        dynamic_world = torch.ones(num_timesteps) * (dynamic_world_class_amount)

    keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
    mask = mask[:, keep_indices]

    if normalize:
        # normalize includes x = x[:, keep_indices]
        x = cast(torch.Tensor, S1_S2_ERA5_SRTM.normalize(x))
        if s2_bands is not None and "B8" in s2_bands and "B4" in s2_bands:
            mask[:, NORMED_BANDS.index("NDVI")] = 0
    else:
        x = x[:, keep_indices]
    # set all masked values to 0.0
    x[mask == 1] = 0.0
    return x, mask, dynamic_world
