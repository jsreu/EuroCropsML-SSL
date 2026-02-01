from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig

from eurocropsssl.dataset.config import ALL_BANDS, ERA5_BANDS, S1_BANDS, S2_BANDS
from eurocropsssl.dataset.eurocrops.dataset import EuroCropsDatasetSSL


@pytest.fixture
def preprocess_config() -> EuroCropsDatasetPreprocessConfig:
    return EuroCropsDatasetPreprocessConfig(
        raw_data_dir=Path("raw_data_dir"), preprocess_dir=Path("preprocess_dir")
    )


@pytest.fixture
def leap_test_arrays_dict() -> dict[str, dict[str, np.ndarray]]:
    return {
        "data": {"S1": np.ones((366, 2)), "S2": np.ones((366, 13))},
        "dates": {
            "S1": pd.date_range(start="2020-01-01", end="2020-12-31").to_numpy(),
            "S2": pd.date_range(start="2020-01-01", end="2020-12-31").to_numpy(),
        },
    }


@pytest.fixture
def test_arrays_dict() -> dict[str, dict[str, np.ndarray]]:
    return {
        "data": {"S1": np.ones((365, 2)), "S2": np.ones((366, 13))},
        "dates": {
            "S1": pd.date_range(start="2021-01-01", end="2021-12-31").to_numpy(),
            "S2": pd.date_range(start="2021-01-01", end="2021-12-31").to_numpy(),
        },
    }


def test_format_dates_day(
    test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDatasetSSL._format_dates(
        date_type="day",
        arrays_dict=test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == test_arrays_dict["data"][satellite].shape
        assert np.max(dates) == 364
        assert np.min(dates) == 0
        assert np.array_equal(np.unique(dates), dates)


def test_format_dates_day_leapyear(
    leap_test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDatasetSSL._format_dates(
        date_type="day",
        arrays_dict=leap_test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == leap_test_arrays_dict["data"][satellite].shape
        assert np.max(dates) == 365
        assert np.min(dates) == 0
        assert np.array_equal(np.unique(dates), dates)


def test_format_dates_month(
    test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDatasetSSL._format_dates(
        date_type="month",
        arrays_dict=test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == (12, test_arrays_dict["data"][satellite].shape[1])
        assert np.max(dates) == 11
        assert np.min(dates) == 0


class TestEuroCropsDatasetSSL:
    @pytest.fixture
    def mock_dataset(self) -> MagicMock:
        """Create a mock EuroCropsDatasetSSL instance."""
        dataset = MagicMock(spec=EuroCropsDatasetSSL)

        # configure mock attributes
        dataset.s1_data_bands = S1_BANDS
        dataset.s2_data_bands = S2_BANDS
        dataset.era5_channels = ERA5_BANDS
        dataset.model_channels = ALL_BANDS
        dataset.padding_value = 0.0

        # test method
        dataset._construct_masked_input = EuroCropsDatasetSSL._construct_masked_input.__get__(
            dataset
        )

        return dataset

    @pytest.fixture
    def mock_presto_dataset(self) -> MagicMock:
        """Create a mock EuroCropsDatasetSSL instance for Presto."""
        dataset = MagicMock(spec=EuroCropsDatasetSSL)

        # configure mock attributes
        dataset.s1_data_bands = S1_BANDS
        dataset.s2_data_bands = [band for band in S2_BANDS if band not in ["01", "09", "10"]]
        dataset.era5_channels = ERA5_BANDS
        dataset.model_channels = ALL_BANDS
        dataset.padding_value = -999.0

        # test method
        dataset._construct_presto_input = EuroCropsDatasetSSL._construct_presto_input.__get__(
            dataset
        )

        return dataset

    def test_construct_masked_input_all_data_sources(self, mock_dataset: MagicMock) -> None:
        """Test _construct_masked_input with all data sources (S1, S2, ERA5, NDVI)."""
        # create sample tensor data with appropriate dimensions
        num_timesteps = 5
        num_channels = len(ALL_BANDS)

        # create sample data with some zeros (padding)
        tensor_data = torch.rand(num_timesteps, num_channels)
        # Add some padding zeros to test masking
        tensor_data[1, 0:2] = 0.0  # S1 padding in timestep 1
        tensor_data[3, 5:7] = 0.0  # S2 padding in timestep 3

        # call the method
        x, mask = mock_dataset._construct_masked_input(tensor_data, ndvi_flag=False)

        # verify output dimensions
        assert x.shape == (num_timesteps, len(mock_dataset.model_channels))
        assert mask.shape == (num_timesteps, len(mock_dataset.model_channels))

        # verify mask has correct values (1.0 where data is zero, 0.0 otherwise)
        assert torch.all(mask[1, 0:2] == 1.0)  # S1 padding in timestep 1
        assert torch.all(mask[3, 5:7] == 1.0)  # S2 padding in timestep 3

        # verify data transfer
        assert torch.allclose(x[0], tensor_data[0])  # first timestep should be unchanged
        assert torch.all(x[1, 0:2] == 0.0)  # padded values should be zero

        # the original data should be correctly transferred to the output tensor
        # for non-zero values, they should match the original data
        s1_indices = list(range(len(mock_dataset.s1_data_bands)))
        s2_indices = list(
            range(
                len(mock_dataset.s1_data_bands),
                len(mock_dataset.s1_data_bands) + len(mock_dataset.s2_data_bands),
            )
        )
        era5_indices = list(
            range(
                len(mock_dataset.s1_data_bands) + len(mock_dataset.s2_data_bands),
                len(mock_dataset.s1_data_bands)
                + len(mock_dataset.s2_data_bands)
                + len(mock_dataset.era5_channels),
            )
        )

        # check non-padded values were transferred correctly
        assert torch.allclose(x[0, s1_indices], tensor_data[0, s1_indices])
        assert torch.allclose(x[0, s2_indices], tensor_data[0, s2_indices])
        assert torch.allclose(x[0, era5_indices], tensor_data[0, era5_indices])

    def test_construct_masked_input_s2_only(self, mock_dataset: MagicMock) -> None:
        """Test _construct_masked_input with only S2 data."""
        # modify the mock to have only S1 data
        mock_dataset.s1_data_bands = None
        mock_dataset.era5_channels = None
        mock_dataset.model_channels = (
            mock_dataset.s2_data_bands.copy()
        )  # use same bands for model channels

        num_timesteps = 4
        tensor_data = torch.rand(num_timesteps, len(mock_dataset.s2_data_bands))
        tensor_data[2, 1] = 0.0  # add a zero value for testing masking

        x, mask = mock_dataset._construct_masked_input(tensor_data, ndvi_flag=False)

        assert x.shape == (num_timesteps, len(mock_dataset.model_channels))
        assert mask.shape == (num_timesteps, len(mock_dataset.model_channels))
        # verify mask has 1.0 at positions with zero values
        assert mask[2, 1] == 1.0  # should mask the zero value
        # verify non-zero data is correctly transferred
        assert torch.allclose(x[0], tensor_data[0])  # first timestep should match
        assert x[2, 1] == 0.0  # zero value should remain zero

    def test_construct_masked_input_with_ndvi(self, mock_dataset: MagicMock) -> None:
        """Test _construct_masked_input with NDVI included."""
        # add NDVI to model channels
        mock_dataset.model_channels = (
            mock_dataset.s1_data_bands
            + mock_dataset.s2_data_bands
            + mock_dataset.era5_channels
            + ["NDVI"]
        )

        num_timesteps = 7
        num_channels = (
            len(mock_dataset.s1_data_bands)
            + len(mock_dataset.s2_data_bands)
            + len(mock_dataset.era5_channels)
            + 1  # add one for NDVI
        )

        tensor_data = torch.rand(num_timesteps, num_channels)
        # last channel is NDVI
        tensor_data[:, -1] = 0.7  # set NDVI values
        tensor_data[1, -1] = 0.0  # set one NDVI value to zero for masking test

        x, mask = mock_dataset._construct_masked_input(tensor_data, ndvi_flag=True)

        assert x.shape == (num_timesteps, len(mock_dataset.model_channels))
        assert mask.shape == (num_timesteps, len(mock_dataset.model_channels))
        # check NDVI mask (1.0 for zero values)
        assert mask[1, -1] == 1.0  # zero NDVI should be masked
        # check NDVI values
        assert x[0, -1] == 0.7  # non-zero NDVI should be preserved
        assert x[1, -1] == 0.0  # zero NDVI should remain zero

        # verify other data is correctly transferred
        s1_indices = list(range(len(mock_dataset.s1_data_bands)))
        assert torch.allclose(x[0, s1_indices], tensor_data[0, s1_indices])

    def test_construct_masked_input_invalid_channels(self, mock_dataset: MagicMock) -> None:
        """Test _construct_masked_input with incorrect number of channels."""
        # Set up a scenario where tensor_data has fewer channels than expected
        mock_dataset.model_channels = mock_dataset.s2_data_bands + mock_dataset.era5_channels

        tensor_data = torch.rand(3, 13)  # 13 channels for S2, ERA5 missing

        # should raise AssertionError because model_channels has more channels than tensor_data
        with pytest.raises(AssertionError):
            mock_dataset._construct_masked_input(tensor_data, ndvi_flag=False)

    def test_construct_presto_input_with_s1_and_s2(self, mock_presto_dataset: MagicMock) -> None:
        """Test _construct_presto_input with both S1 and S2 data."""

        num_timesteps = 5

        # set era5_channels to None
        mock_presto_dataset.era5_channels = None

        s1_channels = len(mock_presto_dataset.s1_data_bands)
        s2_channels = len(mock_presto_dataset.s2_data_bands)

        # create tensor with S1 and S2 data
        tensor_data = torch.rand(num_timesteps, s1_channels + s2_channels)

        # mock presto.construct_single_presto_input to avoid external dependencies
        expected_x = torch.rand(num_timesteps, 10)  # Mock return value
        expected_mask = torch.ones(num_timesteps, 1)
        expected_dw = torch.zeros(num_timesteps, 10)

        with patch(
            "eurocropsssl.dataset.eurocrops.dataset.construct_single_presto_input"
        ) as mock_presto:
            mock_presto.return_value = (expected_x, expected_mask, expected_dw)

            # call the method
            x, extra_meta_data = mock_presto_dataset._construct_presto_input(tensor_data)

            # check the method was called with correct parameters
            mock_presto.assert_called_once()

            # extract the arguments
            _, kwargs = mock_presto.call_args

            # verify that s1 tensor was passed correctly
            assert torch.allclose(kwargs["s1"], tensor_data[:, :s1_channels])

            # verify that s2 tensor was passed correctly
            assert torch.allclose(kwargs["s2"], tensor_data[:, s1_channels:])

            # verify s1_bands parameter was correct
            assert kwargs["s1_bands"] == mock_presto_dataset.s1_data_bands

            # verify s2_bands were converted to the expected format (B1 instead of 01)
            expected_s2_bands = [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B11",
                "B12",
            ]

            assert kwargs["s2_bands"] == expected_s2_bands

            # verify that return values are correct
            assert torch.equal(x, expected_x)
            assert torch.equal(extra_meta_data["presto_mask"], expected_mask)
            assert torch.equal(extra_meta_data["dynamic_world"], expected_dw.to(torch.int))

    def test_construct_presto_input_s1_only(self, mock_presto_dataset: MagicMock) -> None:
        """Test _construct_presto_input with only S1 data."""
        num_timesteps = 4
        s1_channels = len(mock_presto_dataset.s1_data_bands)

        # set s2_data_bands to None
        mock_presto_dataset.s2_data_bands = None

        # create tensor with S1 data only
        tensor_data = torch.rand(num_timesteps, s1_channels)

        # mock presto function
        expected_x = torch.rand(num_timesteps, 8)
        expected_mask = torch.ones(num_timesteps, 1)
        expected_dw = torch.zeros(num_timesteps, 8)

        with patch(
            "eurocropsssl.dataset.eurocrops.dataset.construct_single_presto_input"
        ) as mock_presto:
            mock_presto.return_value = (expected_x, expected_mask, expected_dw)

            # call the method
            x, extra_meta_data = mock_presto_dataset._construct_presto_input(tensor_data)

            # check the method was called with correct parameters
            mock_presto.assert_called_once()

            # Extract the arguments
            _, kwargs = mock_presto.call_args

            # verify that s1 tensor was passed correctly
            assert torch.allclose(kwargs["s1"], tensor_data)

            # verify that s2 is None
            assert kwargs["s2"] is None

            # verify results
            assert torch.equal(x, expected_x)
            assert torch.equal(extra_meta_data["presto_mask"], expected_mask)
            assert torch.equal(extra_meta_data["dynamic_world"], expected_dw.to(torch.int))

    def test_construct_presto_input_s2_only(self, mock_presto_dataset: MagicMock) -> None:
        """Test _construct_presto_input with only S2 data."""
        num_timesteps = 3

        # set s1_data_bands and era5_channels to None
        mock_presto_dataset.s1_data_bands = None
        mock_presto_dataset.era5_channels = None

        s2_channels = len(mock_presto_dataset.s2_data_bands)

        # create tensor with S2 data only
        tensor_data = torch.rand(num_timesteps, s2_channels)

        # mock presto function
        expected_x = torch.rand(num_timesteps, 6)
        expected_mask = torch.ones(num_timesteps, 1)
        expected_dw = torch.zeros(num_timesteps, 6)

        with patch(
            "eurocropsssl.dataset.eurocrops.dataset.construct_single_presto_input"
        ) as mock_presto:
            mock_presto.return_value = (expected_x, expected_mask, expected_dw)

            # call the method
            x, extra_meta_data = mock_presto_dataset._construct_presto_input(tensor_data)

            # check the method was called with correct parameters
            mock_presto.assert_called_once()

            # extract the arguments
            _, kwargs = mock_presto.call_args

            # verify that s1 is None
            assert kwargs["s1"] is None

            # verify that s2 tensor was passed correctly
            assert torch.allclose(kwargs["s2"], tensor_data)

            # verify s2_bands were converted to the expected format (B1 instead of 01)
            expected_s2_bands = [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B11",
                "B12",
            ]
            assert kwargs["s2_bands"] == expected_s2_bands

            # verify results
            assert torch.equal(x, expected_x)
            assert torch.equal(extra_meta_data["presto_mask"], expected_mask)
            assert torch.equal(extra_meta_data["dynamic_world"], expected_dw.to(torch.int))

    def test_construct_presto_input_band_naming(self, mock_presto_dataset: MagicMock) -> None:
        """Test that S2 band names are correctly converted from '01' format to 'B1' format."""
        num_timesteps = 2

        # Use different S2 band names to test conversion
        mock_presto_dataset.s1_data_bands = None
        mock_presto_dataset.s2_data_bands = ["01", "02", "8A", "10", "12"]
        s2_channels = len(mock_presto_dataset.s2_data_bands)

        # create tensor with S2 data only
        tensor_data = torch.rand(num_timesteps, s2_channels)

        # mock presto function
        with patch(
            "eurocropsssl.dataset.eurocrops.dataset.construct_single_presto_input"
        ) as mock_presto:
            mock_presto.return_value = (
                torch.rand(2, 5),
                torch.ones(2, 1),
                torch.zeros(2, 5),
            )

            # call the method
            mock_presto_dataset._construct_presto_input(tensor_data)

            # extract the arguments
            _, kwargs = mock_presto.call_args

            # verify that band names are correctly converted
            expected_s2_bands = ["B1", "B2", "B8A", "B10", "B12"]
            assert kwargs["s2_bands"] == expected_s2_bands
