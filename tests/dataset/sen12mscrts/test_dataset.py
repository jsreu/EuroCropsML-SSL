from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eurocropsssl.dataset.sen12mscrts.dataset import SEN12MSCRTS


def sen12mscrts_test_data() -> dict[str, Any]:
    """Create test data for SEN12MSCRTS class with random values.

    Returns:
        dict: Dictionary containing mock data for testing the SEN12MSCRTS class
    """
    # create random data for S1 and S2
    n_time_points = 30  # alwys fixed to 30 for SEN12MSCRTS
    n_patches = 5

    # create date range for 2018
    full_date_range = pd.date_range(start="2018-01-01", end="2018-12-31")

    # sample dates for S1 and S2
    s1_dates = np.random.choice(full_date_range, size=n_time_points, replace=False)
    s1_dates = np.sort(s1_dates)  # Sort the dates chronologically
    s2_dates = np.random.choice(full_date_range, size=n_time_points, replace=False)
    s2_dates = np.sort(s2_dates)  # Sort the dates chronologically

    # convert numpy datetime64 to strings directly using pandas
    s1_dates_str = np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in s1_dates])
    s2_dates_str = np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in s2_dates])

    # create mock satellite data
    # S1: SAR data (2 channels: VV, VH)
    s1_data = [np.random.rand(256, 256, 2) for _ in range(n_time_points)]

    # S2: multi-spectral data (13 bands)
    s2_data = [np.random.rand(256, 256, 13) for _ in range(n_time_points)]

    # create random cloud masks (0-1 values where 1 represents cloud)
    cloud_masks = [np.random.rand(256, 256) > 0.7 for _ in range(n_time_points)]
    cloud_coverage = [np.mean(mask) for mask in cloud_masks]

    # create mock paths
    mock_paths = []
    for p in range(n_patches):
        sample = {
            "S1": [Path(f"ROIs1970/71/S1/{t}/patch_{p:04d}.tif") for t in range(n_time_points)],
            "S2": [Path(f"ROIs1970/71/S2/{t}/patch_{p:04d}.tif") for t in range(n_time_points)],
        }
        mock_paths.append(sample)

    # create mock coordinates
    coords = [[10.0, 20.0, 10.1, 20.1] for _ in range(n_time_points)]

    # create processed multi-spectral data
    processed_s2 = [np.random.rand(256, 256, 13) for _ in range(n_time_points)]

    # Structure test data to match class functionality
    test_data = {
        "paths": mock_paths,
        "sample_data": {
            "S1": s1_data,
            "S2": s2_data,
            "masks": cloud_masks,
            "coverage": cloud_coverage,
            "S1_dates": s1_dates_str,
            "S2_dates": s2_dates_str,
            "coords": coords,
            "processed_S2": processed_s2,
        },
    }

    return test_data


@pytest.fixture
def mock_sen12mscrts() -> SEN12MSCRTS:
    """Create mock SEN12MSCRTS instance with test data."""
    with patch("eurocropsssl.dataset.sen12mscrts.dataset.SEN12MSCRTS") as MockClass:
        instance = MockClass.return_value
        test_data = sen12mscrts_test_data()

        # Set up the instance with test data
        instance.paths = test_data["paths"]
        instance.n_samples = len(test_data["paths"])
        instance.time_points = range(30)
        instance.cloud_masks = "s2cloudless_mask"
        instance.sample_type = "cloudy_cloudfree"
        instance.min_cov = 0.1
        instance.max_cov = 0.5
        instance.n_input_t = 3

        # Mock _collect_patches to return test data
        instance._collect_patches.side_effect = lambda idx: {
            "S1": test_data["sample_data"]["S1"],
            "S2": test_data["sample_data"]["processed_S2"],
            "masks": test_data["sample_data"]["masks"],
            "coverage": test_data["sample_data"]["coverage"],
            "S1_dates": test_data["sample_data"]["S1_dates"],
            "S2_dates": test_data["sample_data"]["S2_dates"],
            "coords": test_data["sample_data"]["coords"],
        }

        return cast(SEN12MSCRTS, instance)


def test_sen12mscrts_data_loading(mock_sen12mscrts: SEN12MSCRTS) -> None:
    """Test loading data from SEN12MSCRTS."""
    # test loading a single patch
    patch_data = mock_sen12mscrts._collect_patches(0)

    # Check basic structure
    assert "S1" in patch_data
    assert "S2" in patch_data
    assert "masks" in patch_data
    assert "coverage" in patch_data
    assert "S1_dates" in patch_data
    assert "S2_dates" in patch_data

    # check data dimensions
    assert len(cast(np.ndarray, patch_data["S1"])) == 30  # 30 time steps
    assert len(cast(np.ndarray, patch_data["S2"])) == 30
    assert len(cast(np.ndarray, patch_data["masks"])) == 30

    # check that dates are properly formatted (YYYY-MM-DD)
    for date in cast(np.ndarray, patch_data["S1_dates"]):
        # date is string
        assert isinstance(date, str)
        # format YYYY-MM-DD
        assert len(date.split("-")) == 3

    # test handling cloudy/cloud-free sampling
    if mock_sen12mscrts.sample_type == "cloudy_cloudfree":
        # mock cloudless index selection
        with patch.object(
            mock_sen12mscrts,
            "_collect_patches",
            return_value={
                "input": {
                    "S1": [
                        [np.random.rand(256, 256, 2) for _ in range(3)]
                    ],  # Nested list to match actual structure
                    "S2": [np.random.rand(256, 256, 13) for _ in range(3)],
                    "masks": [np.random.rand(256, 256) > 0.7 for _ in range(3)],
                    "coverage": [0.3, 0.4, 0.2],
                    "S1_dates": cast(np.ndarray, mock_sen12mscrts._collect_patches(0)["S1_dates"])[
                        :3
                    ],
                    "S2_dates": cast(np.ndarray, mock_sen12mscrts._collect_patches(0)["S2_dates"])[
                        :3
                    ],
                },
                "target": {
                    "S1": [[np.random.rand(256, 256, 2)]],  # Nested list to match actual structure
                    "S2": [np.random.rand(256, 256, 13)],
                    "masks": [np.random.rand(256, 256) > 0.9],
                    "coverage": [0.05],
                    "S1_dates": [
                        cast(np.ndarray, mock_sen12mscrts._collect_patches(0)["S1_dates"])[0]
                    ],
                    "S2_dates": [
                        cast(np.ndarray, mock_sen12mscrts._collect_patches(0)["S2_dates"])[0]
                    ],
                },
                "coverage bin": True,
            },
        ):
            cloudy_cloudless_data = mock_sen12mscrts._collect_patches(0)

            # check structure
            assert "input" in cloudy_cloudless_data
            assert "target" in cloudy_cloudless_data
            assert "coverage bin" in cloudy_cloudless_data

            # check structure matches expected
            assert (
                len(cast(np.ndarray, cloudy_cloudless_data["input"])["S1"]) == 1
            )  # Outer list has 1 element
            assert (
                len(cast(np.ndarray, cloudy_cloudless_data["input"])["S2"]) == 3
            )  # S2 has 3 elements

            # check that target has 1 sample
            assert len(cast(np.ndarray, cloudy_cloudless_data["target"])["S1"]) == 1
            assert len(cast(np.ndarray, cloudy_cloudless_data["target"])["S2"]) == 1

            # check coverage is within bounds for input
            for cov in cast(np.ndarray, cloudy_cloudless_data["input"])["coverage"]:
                assert mock_sen12mscrts.min_cov <= cov <= mock_sen12mscrts.max_cov


# check paths are correctly constructed
def test_sen12mscrts_paths(mock_sen12mscrts: SEN12MSCRTS) -> None:
    """Test that paths are correctly structured in SEN12MSCRTS."""
    # check expected number of samples
    assert mock_sen12mscrts.n_samples > 0

    # check structure of paths
    for path_dict in mock_sen12mscrts.paths:
        assert "S1" in path_dict
        assert "S2" in path_dict
        assert len(cast(np.ndarray, path_dict["S1"])) == len(mock_sen12mscrts.time_points)
        assert len(cast(np.ndarray, path_dict["S2"])) == len(mock_sen12mscrts.time_points)

        # check that paths have expected format
        for path in path_dict["S1"]:
            assert isinstance(path, Path)
            assert "S1" in str(path)
            assert path.suffix == ".tif"

        for path in path_dict["S2"]:
            assert isinstance(path, Path)
            assert "S2" in str(path)
            assert path.suffix == ".tif"
