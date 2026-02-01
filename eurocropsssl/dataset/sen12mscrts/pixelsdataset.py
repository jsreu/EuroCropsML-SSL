import os
from pathlib import Path
from typing import cast

import h5py
import torch
from eurocropsml.dataset.base import DataItem, LabelledData
from torch.utils.data import Dataset

from eurocropsssl.dataset.config import S1_BANDS, S2_BANDS
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.utils import _pad_missing_dates
from eurocropsssl.settings import EPSILON


class SEN12MSCRTSPixelDataset(Dataset):
    """PyTorch dataset class for the SEN12MSCRTS dataset.

    Making it available for pixel-wise SSL-pretraining.

    Args:
        folder_list: List of paths to parent folders of HDF5 files.
        config: SEN12MS-CR-TS dataset config.
        model_channels: List of used data bands/channels (e.g. S1, S2, ERA5) in the network.
            Only used for potential NDVI calculation.
        padding_value: Padding value used for padding data sequences.

    """

    def __init__(
        self,
        folder_list: list[Path],
        config: SEN12MSCRTSDatasetConfig,
        model_channels: list[str] | None,
        padding_value: float = 0.0,
    ):
        self.folder_list = folder_list
        self.config = config
        self.model_channels = model_channels
        self.padding_value = padding_value

        self._file_mapping = []
        for folder in self.folder_list:
            with os.scandir(folder) as entries:
                folder_files = [(folder, entry.name) for entry in entries]
            self._file_mapping.extend(folder_files)
        self.total_files = len(self._file_mapping)

    def __len__(self) -> int:
        return self.total_files

    def __getitem__(self, idx: int) -> LabelledData:

        # Get folder and filename from the mapping
        folder, filename = self._file_mapping[idx]
        file_path = os.path.join(
            folder, filename
        )  # using os since it's faster than pathlib for joining

        meta_data: dict[str, torch.Tensor] = {
            "dates": torch.tensor([], dtype=torch.int),
            "location": torch.tensor([], dtype=torch.float),
        }

        with h5py.File(file_path, "r") as hdf5_patch:
            # Load HDF5 patch
            tensor_data = torch.from_numpy(hdf5_patch["data"][:])  # [timesteps, channels]
            meta_data["dates"] = torch.from_numpy(
                hdf5_patch["dates"][:]
            )  # Shape: [timesteps, num_sats] ([timesteps] if num_sats==1)
            meta_data["center"] = torch.from_numpy(hdf5_patch["location"][:])  # [2]
            era5 = torch.from_numpy(hdf5_patch["era5"][:])  # [unique_timesteps, num_era5_variables]
            nan_mask = torch.isnan(era5).any(dim=-1, keepdim=True)
            era5 = era5.masked_fill(nan_mask, self.padding_value)

        if self.config.use_sentinel1_data is True:
            # if both S1 and S2 are used
            # removing num_sats dimension to get unique dates
            all_dates: torch.Tensor = meta_data["dates"].flatten()  # [timesteps*num_sats]

            unique_sorted_dates: torch.Tensor = torch.unique(all_dates, sorted=True)

            tensor_data = _pad_missing_dates(  # [unique_timesteps, channels]
                tensor_data,
                meta_data["dates"],
                unique_sorted_dates,
                len(S1_BANDS),  # SEN12MS-CR-TS always uses all S1 bands (if S1 is used)
                len(S2_BANDS),  # SEN12MS-CR-TS always uses all S2 bands
                self.padding_value,
            )

            meta_data["dates"] = unique_sorted_dates

        tensor_data = torch.cat([tensor_data, era5], dim=-1)

        if self.model_channels is not None:
            num_bands = (
                len(self.model_channels)
                if "NDVI" not in self.model_channels
                else len(self.model_channels) - 1
            )
            assert (
                tensor_data.size(-1) == num_bands
            ), "The bands defined in the model config do not match the available data."
            if "NDVI" in self.model_channels:
                # for SEN12MS-CR-TS, we assume that all S2 bands are always used
                s2_bands_indices = [self.model_channels.index(band) for band in S2_BANDS]
                s2_bands = tensor_data[:, s2_bands_indices]
                s2_mask = (s2_bands == 0.0).all(dim=-1)  # padding_value=0.0

                band8 = tensor_data[:, self.model_channels.index("08")]
                band4 = tensor_data[:, self.model_channels.index("04")]
                ndvi = (band8 - band4) / (band8 + band4 + EPSILON)  # if S2 not available, this is 0
                # scale to (0,1]
                ndvi = (ndvi + 1 + EPSILON) / 2
                # set back to zero (no valid data) where there is no S2 data
                ndvi[s2_mask] = 0.0

                tensor_data = torch.cat([tensor_data, ndvi.unsqueeze(-1)], dim=-1)

        target = tensor_data

        return LabelledData(
            DataItem(data=tensor_data, meta_data=cast(dict[str, torch.Tensor], meta_data)),
            target,
        )
