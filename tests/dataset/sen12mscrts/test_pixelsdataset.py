from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from eurocropsml.dataset.base import LabelledData

from eurocropsssl.dataset.config import ALL_BANDS, S1_BANDS, S2_BANDS
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.pixelsdataset import SEN12MSCRTSPixelDataset
from eurocropsssl.settings import EPSILON


# create a fixture for temporary HDF5 files
@pytest.fixture
def mock_data() -> dict[str, np.ndarray | int]:
    """Create basic mock data for testing."""
    n_timesteps = 10
    n_s1_bands = len(S1_BANDS)  # 2 bands: VV, VH
    n_s2_bands = len(S2_BANDS)  # 13 bands
    n_era5_vars = 2  # number of ERA5 variables (temperature & precipitation)

    # create mock data tensors
    data = np.random.rand(n_timesteps, n_s1_bands + n_s2_bands).astype(np.float32)
    dates = np.array([[i * 10, i * 10 + 2] for i in range(n_timesteps)], dtype=np.int32)
    location = np.array([45.0, 10.0], dtype=np.float32)  # lat, lon

    # calculate unique dates and prepare matching ERA5 data
    unique_dates = np.unique(dates.flatten())
    n_unique_dates = len(unique_dates)
    era5 = np.random.rand(n_unique_dates, n_era5_vars).astype(np.float32)

    # overwrite band 4 and band 8 to later test NDVI
    band4_idx = n_s1_bands + S2_BANDS.index("04")
    band8_idx = n_s1_bands + S2_BANDS.index("08")
    data[:, band4_idx] = 0.1
    data[:, band8_idx] = 0.5

    return {
        "data": data,
        "dates": dates,
        "location": location,
        "era5": era5,
        "n_timesteps": n_timesteps,
        "n_s1_bands": n_s1_bands,
        "n_s2_bands": n_s2_bands,
        "n_era5_vars": n_era5_vars,
        "n_unique_dates": n_unique_dates,
        "unique_dates": unique_dates,
    }


@pytest.fixture
def mock_h5py_file(mock_data: dict[str, np.ndarray | int]) -> MagicMock:
    """Create a mock HDF5 file using the mock data."""
    # create mock h5py file object
    mock_file = MagicMock(spec=h5py.File)
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None

    # sSet up mock datasets with data
    mock_data_ds = MagicMock()
    mock_data_ds.__getitem__.return_value = mock_data["data"]

    mock_dates_ds = MagicMock()
    mock_dates_ds.__getitem__.return_value = mock_data["dates"]

    mock_location_ds = MagicMock()
    mock_location_ds.__getitem__.return_value = mock_data["location"]

    mock_era5_ds = MagicMock()
    mock_era5_ds.__getitem__.return_value = mock_data["era5"]

    # set up dictionary-like access to datasets
    mock_file.__getitem__.side_effect = lambda key: {
        "data": mock_data_ds,
        "dates": mock_dates_ds,
        "location": mock_location_ds,
        "era5": mock_era5_ds,
    }[key]

    return mock_file


@pytest.fixture
def mock_dataset_config() -> MagicMock:
    """Create a mock dataset config."""
    config = MagicMock(spec=SEN12MSCRTSDatasetConfig)
    config.use_sentinel1_data = True
    return config


@pytest.fixture
def mock_folder_structure(tmp_path: Path) -> list[Path]:
    """Create a temporary folder structure with mock HDF5 files."""
    folder1 = tmp_path / "folder1"
    folder2 = tmp_path / "folder2"
    folder1.mkdir()
    folder2.mkdir()

    for i in range(3):
        (folder1 / f"patch_{i}.h5").touch()
    for i in range(2):
        (folder2 / f"patch_{i}.h5").touch()

    return [folder1, folder2]


# Main test function for SEN12MSCRTSPixelDataset
def test_sen12mscrts_pixel_dataset(
    mock_data: dict[str, np.ndarray | int],
    mock_h5py_file: MagicMock,
    mock_dataset_config: MagicMock,
    mock_folder_structure: list[Path],
) -> None:
    """Test basic functionality of the SEN12MSCRTSPixelDataset class."""
    # create padded tensor with correct dimensions for _pad_missing_dates
    padded_tensor = torch.zeros((20, len(S1_BANDS) + len(S2_BANDS)), dtype=torch.float32)
    # mock h5py.File to return mock file
    with (
        patch(
            "eurocropsssl.dataset.sen12mscrts.utils._pad_missing_dates",
            return_value=padded_tensor,
        ),
        patch("h5py.File", return_value=mock_h5py_file),
    ):
        # Create the dataset
        dataset = SEN12MSCRTSPixelDataset(
            folder_list=mock_folder_structure,
            config=mock_dataset_config,
            model_channels=ALL_BANDS,  # Use all bands to match data dimensions
            padding_value=0.0,
        )

        # test dataset length
        assert len(dataset) == 5  # 3 files in folder1 + 2 files in folder2

        # test __getitem__
        item = dataset[0]

        # verify returned item is of the correct type
        assert isinstance(item, LabelledData)

        expected_shape = (
            mock_data["n_unique_dates"],
            mock_data["n_s1_bands"] + mock_data["n_s2_bands"] + mock_data["n_era5_vars"],
        )

        # check data shape matches expected shape
        assert item.data.shape == expected_shape
        assert item.label.shape == expected_shape


def test_sen12mscrts_ndvi_calculation(
    mock_data: dict[str, np.ndarray | int],
    mock_h5py_file: MagicMock,
    mock_dataset_config: MagicMock,
    mock_folder_structure: list[Path],
) -> None:
    """Test that NDVI is correctly calculated when requested."""
    # add NDVI to list of band
    model_channels = ALL_BANDS + ["NDVI"]

    # expected NDVI calculation (band4=0.1; band8=0.5)
    expected_ndvi = (0.5 - 0.1) / (0.5 + 0.1 + EPSILON)
    expected_ndvi = (expected_ndvi + 1 + EPSILON) / 2

    with patch("h5py.File", return_value=mock_h5py_file):
        # create dataset with NDVI
        dataset = SEN12MSCRTSPixelDataset(
            folder_list=mock_folder_structure,
            config=mock_dataset_config,
            model_channels=model_channels,
            padding_value=0.0,
        )

        # get an item
        item = dataset[0]

        # check data shape includes the additional NDVI band
        expected_shape = (
            mock_data["n_unique_dates"],
            mock_data["n_s1_bands"]
            + mock_data["n_s2_bands"]
            + mock_data["n_era5_vars"]
            + 1,  # +1 for NDVI
        )

        # check data shape is as expected
        assert item.data.shape == expected_shape

        # Check NDVI values are correct (last channel)
        ndvi_values = item.data[:, -1]

        # Recreate the S2 mask logic
        s2_start = mock_data["n_s1_bands"]
        s2_end = s2_start + mock_data["n_s2_bands"]
        s2_bands = item.data[:, s2_start:s2_end]
        s2_mask = (s2_bands == 0.0).all(dim=-1)

        # NDVI should be zero where S2 is missing
        assert torch.all(ndvi_values[s2_mask] == 0.0)

        # NDVI should match expected NDVI where S2 is valid
        valid_ndvi = ndvi_values[~s2_mask]
        expected_tensor = torch.full_like(valid_ndvi, expected_ndvi)

        assert torch.allclose(valid_ndvi, expected_tensor, rtol=1e-5, atol=1e-5)
