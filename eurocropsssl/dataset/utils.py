from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _format_dates(
    dates: np.ndarray,
) -> np.ndarray:
    dates = pd.to_datetime(dates).day_of_year.to_numpy() - 1
    return dates


def _pad_missing_dates(
    data: torch.Tensor,
    dates: torch.Tensor,
    unique_sorted_dates: torch.Tensor,
    num_channels_1: int = 2,
    num_channels_2: int = 13,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pad missing data with a certain value between two group of channels."""

    # Combined output tensor for S1 and S2 bands with padding values
    combined_data = torch.full(
        (len(unique_sorted_dates), num_channels_1 + num_channels_2),
        padding_value,
    )
    s1_dates = dates[:, 0]  # first column corresponds to S1
    s1_indices = torch.searchsorted(unique_sorted_dates, s1_dates.contiguous())
    s2_dates = dates[:, 1]  # second column corresponds to S2
    s2_indices = torch.searchsorted(unique_sorted_dates, s2_dates.contiguous())

    # first group channels
    combined_data[s1_indices, :num_channels_1] = data[:, :num_channels_1]

    # second group of channels
    combined_data[s2_indices, num_channels_1:] = data[:, num_channels_1:]

    return combined_data
