from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eurocropsssl.dataset.era5.preprocess import (
    MAX_PRECIP,
    MAX_TEMP,
    MIN_TEMP,
    get_era5_values,
)
from eurocropsssl.settings import EPSILON


@pytest.fixture
def mock_era5_data() -> dict[str, xr.Dataset]:
    """Create mock ERA5 data for testing"""
    # create mock analysis dataset with temperature
    ds_an = xr.Dataset(
        data_vars={
            "t2m": (
                ["time", "latitude", "longitude"],
                np.ones((24, 3, 3)) * 280,
            ),  # temperature: 280 Kelvin (about 7°C)
        },
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": np.linspace(40.0, 45.0, 3),
            "longitude": np.linspace(10.0, 15.0, 3),
        },
    )

    # create mock forecast dataset with precipitation
    ds_fc = xr.Dataset(
        data_vars={
            "tp": (
                ["time", "latitude", "longitude"],
                np.ones((24, 3, 3)) * 0.01,
            ),  # precipitation: 0.01m (10mm)
        },
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": np.linspace(40.0, 45.0, 3),
            "longitude": np.linspace(10.0, 15.0, 3),
        },
    )

    return {"analysis": ds_an, "forecast": ds_fc}


@pytest.fixture
def mock_era5_file(
    tmp_path: Path, mock_era5_data: dict[str, xr.Dataset]
) -> dict[str, Path | dict[str, xr.Dataset]]:
    """Create a temporary directory with mocked ERA5 file"""
    test_dir = tmp_path / "era5"
    test_dir.mkdir()
    test_file = test_dir / "era5_2022_05_15.grb"
    test_file.touch()  # Create empty file

    return {"dir": test_dir, "file": test_file, "data": mock_era5_data}


def test_get_era5_values_file_not_found() -> None:
    """Test that FileNotFoundError is raised when ERA5 file is missing"""
    # setup
    date = "2022-05-15"
    bbox = (10.0, 40.0, 15.0, 45.0)
    base_path = Path("/non_existent_path")

    # execute and verify
    with pytest.raises(FileNotFoundError) as exc_info:
        get_era5_values(date, bbox, base_path)

    assert "Data file missing" in str(exc_info.value)


@patch("xarray.open_dataset")
def test_get_era5_values(
    mock_open_dataset: xr.Dataset,
    mock_era5_file: dict[str, Path | dict[str, xr.Dataset]],
) -> None:
    """Test extraction of ERA5 values from file"""
    # setup
    date = "2022-05-15"
    bbox = (10.0, 40.0, 15.0, 45.0)
    base_path = mock_era5_file["dir"]

    # mock the xarray open_dataset to return requested mock datasets
    mock_open_dataset.side_effect = [
        cast(dict[str, xr.Dataset], mock_era5_file["data"])["analysis"],  # return analysis dataset
        cast(dict[str, xr.Dataset], mock_era5_file["data"])["forecast"],  # return forecast dataset
    ]

    # mock the context manager behavior
    mock_open_dataset.return_value.__enter__.side_effect = [
        cast(dict[str, xr.Dataset], mock_era5_file["data"])["analysis"],  # return analysis dataset
        cast(dict[str, xr.Dataset], mock_era5_file["data"])["forecast"],  # return forecast dataset
    ]

    result = get_era5_values(date, bbox, cast(Path, base_path), normalize=True)

    assert len(result) == 2  # Should have temperature and precipitation values

    # check temperature normalization (t2m)
    t2m_normalized = (280 - 239) / (324 - 239) * (1 - EPSILON) + EPSILON
    assert pytest.approx(result[0], abs=1e-5) == t2m_normalized

    # check precipitation normalization (tp)
    # total precipitation should be clipped to MAX_PRECIP (0.03m)
    tp_normalized = (0.03 / 0.03) * (1 - EPSILON) + EPSILON
    assert pytest.approx(result[1], abs=1e-5) == tp_normalized


@patch("xarray.open_dataset")
def test_get_era5_values_interpolation(
    mock_open_dataset: xr.Dataset,
    mock_era5_file: dict[str, Path | dict[str, xr.Dataset]],
) -> None:
    """Test that values are properly interpolated for the center of the bbox"""
    date = "2022-05-15"
    # use coordinates that require interpolation
    bbox = (11.0, 41.0, 14.0, 44.0)  # Center: 12.5, 42.5
    base_path = mock_era5_file["dir"]

    # create data with varying values to test interpolation
    lat_values = np.array([41.0, 42.5, 44.0])
    lon_values = np.array([11.0, 12.5, 14.0])

    # temperature increases with latitude and longitude
    temp_grid = np.zeros((24, 3, 3))
    for i in range(3):
        for j in range(3):
            temp_grid[:, i, j] = 260 + i * 10 + j * 5  # Varying temperature

    # precipitation decreases with latitude and longitude
    precip_grid = np.zeros((24, 3, 3))
    for i in range(3):
        for j in range(3):
            precip_grid[:, i, j] = 0.015 - i * 0.002 - j * 0.001  # Varying precipitation

    # create datasets with these values
    ds_an = xr.Dataset(
        data_vars={"t2m": (["time", "latitude", "longitude"], temp_grid)},
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": lat_values,
            "longitude": lon_values,
        },
    )

    ds_fc = xr.Dataset(
        data_vars={"tp": (["time", "latitude", "longitude"], precip_grid)},
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": lat_values,
            "longitude": lon_values,
        },
    )

    # create context manager mock objects
    mock_an_ctx = MagicMock()
    mock_an_ctx.__enter__.return_value = ds_an

    mock_fc_ctx = MagicMock()
    mock_fc_ctx.__enter__.return_value = ds_fc

    # configure mock_open_dataset to return context manager mocks
    mock_open_dataset.side_effect = [mock_an_ctx, mock_fc_ctx]

    result = get_era5_values(date, bbox, cast(Path, base_path), normalize=True)

    # verify - should interpolate to the center values
    # center temperature (at 12.5, 42.5) should be 275
    center_temp = 275
    t2m_normalized = (center_temp - 239) / (324 - 239) * (1 - EPSILON) + EPSILON

    # this value exceeds MAX_PRECIP (0.03), so it should be clipped
    tp_normalized = (0.03 / 0.03) * (1 - EPSILON) + EPSILON

    assert pytest.approx(result[0], abs=1e-5) == t2m_normalized
    assert pytest.approx(result[1], abs=1e-5) == tp_normalized


@patch("xarray.open_dataset")
def test_get_era5_values_clipping(
    mock_open_dataset: xr.Dataset,
    mock_era5_file: dict[str, Path | dict[str, xr.Dataset]],
) -> None:
    """Test that values are properly clipped to min/max ranges"""
    date = "2022-05-15"
    bbox = (10.0, 40.0, 15.0, 45.0)
    base_path = mock_era5_file["dir"]

    # create extreme values that should be clipped
    ds_an = xr.Dataset(
        data_vars={
            "t2m": (
                ["time", "latitude", "longitude"],
                np.ones((24, 3, 3)) * 400,
            ),  # > 324 (MAX_TEMP), should be clipped to MAX_TEMP
        },
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": np.linspace(40.0, 45.0, 3),
            "longitude": np.linspace(10.0, 15.0, 3),
        },
    )

    ds_fc = xr.Dataset(
        data_vars={
            "tp": (
                ["time", "latitude", "longitude"],
                np.ones((24, 3, 3)) * 0.05,
            ),  # > 0.03 (MAX_PRECIP), should be clipped to MAX_PRECIP
        },
        coords={
            "time": pd.date_range("2022-05-15", periods=24, freq="H"),
            "latitude": np.linspace(40.0, 45.0, 3),
            "longitude": np.linspace(10.0, 15.0, 3),
        },
    )

    # mock the xarray open_dataset to return temp and precip
    mock_open_dataset.side_effect = [ds_an, ds_fc]
    mock_open_dataset.return_value.__enter__.side_effect = [ds_an, ds_fc]

    result = get_era5_values(date, bbox, cast(Path, base_path), normalize=True)

    # temperature should be clipped to MAX_TEMP (324K)
    t2m_normalized = (MAX_TEMP - MIN_TEMP) / (MAX_TEMP - MIN_TEMP) * (1 - EPSILON) + EPSILON
    assert pytest.approx(result[0], abs=1e-5) == t2m_normalized

    # total precipitation should be clipped to MAX_PRECIP (0.03m)
    tp_normalized = (MAX_PRECIP / MAX_PRECIP) * (1 - EPSILON) + EPSILON
    assert pytest.approx(result[1], abs=1e-5) == tp_normalized
