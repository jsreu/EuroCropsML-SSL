import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from eurocropsml.dataset.base import DataItem, LabelledData
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.dataset.dataset import (
    NORMALIZING_FACTOR_S1,
    NORMALIZING_FACTOR_S2,
    EuroCropsDataset,
)
from eurocropsml.dataset.utils import _pad_missing_dates, _unique_dates

from eurocropsssl.dataset.era5.preprocess import EPSILON, MAX_PRECIP, MAX_TEMP, MIN_TEMP
from eurocropsssl.dataset.eurocrops.config import EuroCropsDatasetConfigSSL
from eurocropsssl.dataset.eurocrops.utils import (
    EuroCropsDatasetERA5,
    MMapStoreERA5,
    _format_band_name,
)
from eurocropsssl.models.presto_modules.utils import construct_single_presto_input

ERA5_PRESTO_CLIP_RANGES = {
    "2m_temperature": [250.15, 309.15],
    "total_precipitation": [0.0, 0.03],
}

logger = logging.getLogger(__name__)


class EuroCropsDatasetSSL(EuroCropsDataset):
    """PyTorch dataset class for the EuroCropsML dataset, making it available for SSL-pretraining.

    Args:
        file_dict: Dictionary of file paths (one file per parcel/time series) for each satellite.
        encode: Encoding used to encode the classes into integers.
        mmap_store: Instance of memory map store.
        config: EuroCropsDatasetConfig instance.
        preprocess_config: EuroCropsDatasetPreprocessConfig instance.
        self_supervised: Flag to make dataset usable in a self-supervised setting.
        max_masked_ratio: Ratio of max masked values.
        model_channels: List of used data bands/channels (e.g. S1, S2, ERA5) in the network.
            Only used for potential NDVI calculation.
        presto_input: Whether to use presto input (for the Presto model).
        pretrained. Flag whether the network was pre-trained or whether we fine-tune a random
            initialization.
        ndvi_flag: Flag whether to add NDVI if Sentinel-2 is available. Defaults to True.
        padding_value: Padding value used for padding data sequences.

    """

    def __init__(
        self,
        file_dict: dict[str, list[Path]],
        encode: dict[int, int],
        mmap_store: MMapStoreERA5,
        config: EuroCropsDatasetConfigSSL,
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        self_supervised: bool,
        model_channels: list[str] | None,
        presto_input: bool,
        pretrained: bool,
        ndvi_flag: bool = True,
        padding_value: float = 0.0,
    ):
        super().__init__(
            file_dict,
            encode,
            mmap_store,
            config,
            preprocess_config,
            pad_seq_to_366=False,
            padding_value=padding_value,
        )

        self.self_supervised = self_supervised
        self.presto_input = presto_input
        self.model_channels = model_channels
        self.pretrained = pretrained
        self.ndvi_flag = ndvi_flag

        if "ERA5" in self.config.data_sources:
            self.era5_channels: list[str] | None = self.config.era5_channels
        else:
            self.era5_channels = None

    def _construct_presto_input(
        self,
        tensor_data: torch.Tensor,
        padding_value: float = -999.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s1_tensor = None
        s2_tensor = None
        era5_tensor = None

        if self.s1_data_bands is not None:
            # presto assumes unnormalized S1 values in the range ~ [-50,1]
            s1_tensor = tensor_data[:, : len(self.s1_data_bands)]
        if self.s2_data_bands is not None:
            start_idx = len(self.s1_data_bands) if self.s1_data_bands is not None else 0
            end_idx = tensor_data.shape[1] - (
                len(self.era5_channels) if self.era5_channels is not None else 0
            )
            # this assumes unnormalized S2 values in the range ~ [0, 1e4]
            s2_tensor = tensor_data[:, start_idx:end_idx]
        if self.era5_channels is not None:
            era5_tensor = tensor_data[:, tensor_data.shape[1] - len(self.era5_channels) :]
            for i, ch in enumerate(self.era5_channels):
                vmin, vmax = ERA5_PRESTO_CLIP_RANGES[ch]
                era5_tensor[:, i] = torch.clamp(era5_tensor[:, i], vmin, vmax)

        x, mask, dynamic_world = construct_single_presto_input(
            s1=s1_tensor,
            s1_bands=self.s1_data_bands,
            s2=s2_tensor,
            # Presto requires band names "B1", ..., "B12" instead of "01", ..., "12"
            s2_bands=(
                [_format_band_name(b) for b in self.s2_data_bands]
                if self.s2_data_bands is not None
                else None
            ),
            era5=era5_tensor,
            era5_bands=self.era5_channels,
            padding_value=padding_value,
        )
        extra_meta_data = {
            "dynamic_world": dynamic_world.to(torch.int),
            "presto_mask": mask,
        }
        return x, extra_meta_data

    def _construct_masked_input(
        self,
        tensor_data: torch.Tensor,
        ndvi_flag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Method to mask non-available channels. Adjusted from Presto.

        Args:
            tensor_data: Input tensor of shape (num_timesteps, num_channels).
            ndvi_flag: Whether the tensor_data contains NDVI values.

        The following data is allowed:
        s1: ["VV", "VH"]
        s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        era5: ["2m_temperature", "total_precipitation"]
        ndvi: ["NDVI"]

        Raises:
            AssertionError if the number of defined variables and corresponding band does not
            match the channel dimension of the tensor.

        """
        s1_tensor = None
        s2_tensor = None
        era5_tensor = None
        ndvi_tensor = None
        num_bands = 0

        if self.s1_data_bands is not None:
            # this assumes already padded and scaled values to (0,1]
            s1_tensor = tensor_data[:, : len(self.s1_data_bands)]
            num_bands += len(self.s1_data_bands)
        if self.s2_data_bands is not None:
            # this assumes already padded and scaled values to (0,1]
            s2_tensor = tensor_data[:, num_bands : num_bands + len(self.s2_data_bands)]
            num_bands += len(self.s2_data_bands)
        if self.era5_channels is not None:
            # this assumes already padded and scaled values to (0,1]
            era5_tensor = tensor_data[:, num_bands : num_bands + len(self.era5_channels)]
            num_bands += len(self.era5_channels)
        if ndvi_flag:
            ndvi_tensor = tensor_data[:, num_bands:]
            num_bands += 1

        if tensor_data.size(-1) != num_bands:
            raise AssertionError(
                "The number of defined data sources and corresponding channels in "
                "EuroCropsDatasetConfigSSL is greater than the number of channels in "
                f"the data ({str(tensor_data.size(-1))})."
            )

        num_timesteps_list = [
            x.shape[0] for x in [s1_tensor, s2_tensor, era5_tensor, ndvi_tensor] if x is not None
        ]
        assert len(num_timesteps_list) > 0
        assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
        num_timesteps = num_timesteps_list[0]
        mask, x = torch.ones(
            num_timesteps, len(cast(list, self.model_channels)), dtype=torch.float32
        ), torch.zeros(num_timesteps, len(cast(list, self.model_channels)), dtype=torch.float32)

        for data, input_bands in [
            (s1_tensor, self.s1_data_bands),
            (s2_tensor, self.s2_data_bands),
            (era5_tensor, self.era5_channels),
            (ndvi_tensor, ["NDVI"]),
        ]:

            if data is not None:
                assert input_bands is not None
            else:
                continue

            # construct a mapping from the input bands to the expected bands in the network
            input_to_output_mapping = [
                cast(list, self.model_channels).index(val) for val in input_bands
            ]

            x[:, input_to_output_mapping] = data
            # also mask values that have padding value 0.0 (padding value)
            zero_mask = data == self.padding_value  # shape: (timesteps, selected_channels)
            # Scatter into the correct output positions in mask
            mask[:, input_to_output_mapping] = zero_mask.float()

        return x, mask

    def __getitem__(
        self,
        idx: int,
    ) -> LabelledData:
        f = {key: paths[idx] for key, paths in self.file_dict.items()}
        arrays_dict = self.mmap_store[f]

        arrays_dict = EuroCropsDatasetERA5._format_dates(
            self.preprocess_config,
            self.s2_data_bands,
            self.config.date_type,
            arrays_dict,
            self.padding_value,
        )

        np_data_dict = arrays_dict.pop("data")
        era5_array = arrays_dict.pop("era5")

        era5_tensor = None
        if self.era5_channels is not None:
            # Use the first available satellite key (e.g., 'S1' or 'S2')
            # to access the corresponding dates for ERA5 data.
            satellite_key = next(iter(f))
            era5_tensor = torch.tensor(era5_array[satellite_key], dtype=torch.float)

        if self.keep_band_idxs is not None:
            np_data_dict["S2"] = np_data_dict["S2"][:, self.keep_band_idxs]

        meta_data: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            array_name: {key: torch.tensor(np_array) for key, np_array in meta_array.items()}
            for array_name, meta_array in arrays_dict.items()
        }
        # center is always the same, replace by first value
        meta_data["center"] = next(iter(meta_data["center"].values()))
        # swap to lat, lon
        center_tensor = cast(torch.Tensor, meta_data["center"])
        meta_data["center"] = torch.flip(center_tensor, [0])

        # normalization and scaling to (0,1]
        if self.config.normalize:
            if "S1" in np_data_dict:
                # range before normalization and scaling: [-50.0, 1.0]
                normalized_s1 = (np_data_dict["S1"] + 50.0) * NORMALIZING_FACTOR_S1
                np_data_dict["S1"] = normalized_s1 * (1 - EPSILON) + EPSILON
            if "S2" in np_data_dict:
                # range before normalization and scaling: [0.0, 1.0e+4]
                normalized_s2 = np_data_dict["S2"] * NORMALIZING_FACTOR_S2
                np_data_dict["S2"] = normalized_s2 * (1 - EPSILON) + EPSILON
            if era5_tensor is not None and self.era5_channels is not None:
                for i, ch in enumerate(self.era5_channels):
                    col = era5_tensor[:, i]
                    if ch == "2m_temperature":
                        col = torch.clamp(col, MIN_TEMP, MAX_TEMP)
                        col = (col - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
                        col = col * (1 - EPSILON) + EPSILON
                    elif ch == "total_precipitation":
                        col = torch.clamp(col, 0, MAX_PRECIP)
                        col = (col / MAX_PRECIP) * (1 - EPSILON) + EPSILON
                    era5_tensor[:, i] = col

        # pad missing values if both S1 and S2 are used and not the Presto model
        if len(f) == 2 and not self.presto_input:
            all_dates: torch.Tensor = _unique_dates(
                cast(dict[str, torch.Tensor], meta_data["dates"]), list(f.keys())
            )
            np_data = _pad_missing_dates(
                np_data_dict,
                cast(dict[str, torch.Tensor], meta_data["dates"]),
                all_dates,
                len(self.s1_data_bands),
                len(self.s2_data_bands),
                padding_value=self.padding_value,
            )
            meta_data["dates"] = all_dates  # only keep full range of dates
        else:
            meta_data["dates"] = _unique_dates(
                cast(dict[str, torch.Tensor], meta_data["dates"]), list(f.keys())
            )
            np_data = np.hstack(list(np_data_dict.values()))

        if era5_tensor is not None:
            # some ERA5 data is nan, set to padding value so it gets masked out
            nan_mask = torch.isnan(era5_tensor).any(dim=-1, keepdim=True)
            era5_tensor = era5_tensor.masked_fill(nan_mask, self.padding_value)
            np_era5 = era5_tensor.numpy()
            era5_filtered = np_era5[cast(torch.Tensor, meta_data["dates"]).numpy()]
            np_data = np.concatenate([np_data, era5_filtered], axis=1)

        tensor_data = torch.tensor(np_data, dtype=torch.float)

        if self.model_channels is not None and self.self_supervised:
            num_bands = (
                len(self.model_channels)
                if "NDVI" not in self.model_channels
                else len(self.model_channels) - 1
            )
            assert (
                tensor_data.size(-1) == num_bands
            ), "The bands defined in the model config do not match the available data."

        ndvi_exists: bool = False
        if self.ndvi_flag and not self.presto_input:
            if self.s2_data_bands is not None:
                available_channels = (
                    (self.s1_data_bands or []) + (self.s2_data_bands) + (self.era5_channels or [])
                )
                s2_bands_indices = [available_channels.index(band) for band in self.s2_data_bands]
                s2_bands = tensor_data[:, s2_bands_indices]
                s2_mask = (s2_bands == self.padding_value).all(dim=-1)  # padding_value=0.0

                band8 = tensor_data[:, available_channels.index("08")]
                band4 = tensor_data[:, available_channels.index("04")]
                ndvi = (band8 - band4) / (band8 + band4 + EPSILON)  # if S2 not available, this is 0
                # scale to (0,1]
                ndvi = (ndvi + 1 + EPSILON) / 2
                # set back to zero (no valid data) where there is no S2 data
                ndvi[s2_mask] = self.padding_value
                tensor_data = torch.cat([tensor_data, ndvi.unsqueeze(-1)], dim=-1)
                ndvi_exists = True
            else:
                raise AssertionError(
                    "S2 is not available, hence NDVI cannot be calculated. "
                    "Please set flag `ndvi_flag` to False. "
                )

        if self.presto_input:
            tensor_data, extra_meta_data = self._construct_presto_input(
                tensor_data, self.padding_value
            )
            meta_data.update(extra_meta_data)

        if self.self_supervised:
            target = tensor_data
        else:
            # do not mask for random initialization
            if self.model_channels is not None and self.pretrained:
                tensor_data, mask = self._construct_masked_input(tensor_data, ndvi_exists)
                meta_data["aug_mask"] = mask
            filepath: Path = next(iter(f.values()))
            y = int(filepath.stem.split("_")[-1])
            # encode class
            y = self.encode[y]
            target = torch.tensor(y)

        return LabelledData(DataItem(data=tensor_data, meta_data=meta_data), target)
