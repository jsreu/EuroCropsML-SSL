from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from eurocropsssl.dataset.sen12mscrts.utils import (
    _bbox_to_polygon,
    _format_dates,
    _pad_missing_dates,
    get_cloud_cloudshadow_mask,
    get_cloud_map,
    get_cloud_mask,
    get_shadow_mask,
    normalized_difference,
    process_MS,
    process_SAR,
    rescale,
)
from eurocropsssl.settings import EPSILON


@pytest.fixture
def leap_test_arrays_dict() -> dict[str, np.ndarray | torch.Tensor]:
    """Create test data for leap year with random values."""
    # Create random data for S1 and S2
    s1_data = np.random.rand(150, 2)  # 150 observations for S1
    s2_data = np.random.rand(150, 13)  # 150 observations for S2

    # Create date range for leap year
    full_date_range = pd.date_range(start="2020-01-01", end="2020-12-31")

    # Randomly sample dates (with some overlap and some unique to each satellite)
    s1_dates = np.random.choice(full_date_range, size=150, replace=False)
    s1_dates = np.sort(s1_dates)  # Sort dates

    s2_dates = np.random.choice(full_date_range, size=150, replace=False)
    s2_dates = np.sort(s2_dates)  # Sort dates

    # stack data
    data = torch.zeros(150, 15)
    data[:, :2] = torch.from_numpy(s1_data)
    data[:, 2:] = torch.from_numpy(s2_data)
    dates = np.zeros((150, 2), dtype="datetime64[ns]")
    dates[:, 0] = s1_dates
    dates[:, 1] = s2_dates

    return {
        "data": data,
        "dates": dates,
    }


@pytest.fixture
def non_leap_test_arrays_dict() -> dict[str, np.ndarray | torch.Tensor]:
    """Create test data for non-leap year with random values."""
    # Create random data for S1 and S2
    s1_data = np.random.rand(150, 2)  # 150 observations for S1
    s2_data = np.random.rand(150, 13)  # 150 observations for S2

    # Create date range for non-leap year
    full_date_range = pd.date_range(start="2021-01-01", end="2021-12-31")

    # Randomly sample dates
    s1_dates = np.random.choice(full_date_range, size=150, replace=False)
    s1_dates = np.sort(s1_dates)  # Sort dates

    s2_dates = np.random.choice(full_date_range, size=150, replace=False)
    s2_dates = np.sort(s2_dates)  # Sort dates

    # stack data
    data = torch.zeros(150, 15)
    data[:, :2] = torch.from_numpy(s1_data)
    data[:, 2:] = torch.from_numpy(s2_data)
    dates = np.zeros((150, 2), dtype="datetime64[ns]")
    dates[:, 0] = s1_dates
    dates[:, 1] = s2_dates

    return {
        "data": data,
        "dates": dates,
    }


@pytest.fixture
def test_s1_img() -> np.ndarray:
    img = np.arange(-55, 5, 100, dtype=np.float32)
    return img


@pytest.fixture
def test_s1_img_rescale() -> np.ndarray:
    img = np.array([-55, 5], dtype=np.float32)
    return img


@pytest.fixture
def test_s2_img() -> np.ndarray:
    img = np.arange(-1, 11000, 100, dtype=np.float32)
    return img


@pytest.fixture
def test_cloud_img() -> np.ndarray:
    """Create a test image with cloud patterns."""
    # 13 bands, 10x10 pixels
    img = np.ones((13, 10, 10), dtype=np.float32) * 2000  # Medium brightness everywhere

    # Create cloudy region in top-right corner
    # Clouds are bright in blue (band 2), aerosol (band 1) and cirrus (band 10) bands
    img[0, :5, 5:] = 3000  # Aerosol band 1
    img[1, :5, 5:] = 5000  # Blue band 2
    img[10, :5, 5:] = 6000  # Cirrus band 10

    # Add some brightness in RGB bands (except band 2, since we set this to 5000 already)
    img[2:4, :5, 5:] = 4000  # green (band 3) and red (band 4) bands

    return img


@pytest.fixture
def test_shadow_img() -> np.ndarray:
    """Create a test image with shadow patterns."""
    # 13 bands, 10x10 pixels
    img = np.ones((13, 10, 10), dtype=np.float32) * 2000  # Medium brightness everywhere

    # Create shadow region in bottom-left corner
    # Shadows have low NIR (band 8) and SWIR1 (band 11) values but medium-low blue (band 2) values
    img[7, 5:, :5] = 500  # NIR band 8 - low in shadows
    img[11, 5:, :5] = 400  # SWIR1 band 11 - low in shadows
    img[1, 5:, :5] = 800  # Blue band 2 - medium-low in shadows

    return img


@pytest.fixture
def test_cloud_shadow_img() -> np.ndarray:
    """Create a test image with both cloud and shadow patterns."""
    # 13 bands, 10x10 pixels
    img = np.ones((13, 10, 10), dtype=np.float32) * 2000  # Medium brightness everywhere

    # Create cloudy region in the top-right corner
    img[0, :5, 5:] = 3000  # Aerosol band 1
    img[1, :5, 5:] = 5000  # Blue band 2
    img[10, :5, 5:] = 6000  # Cirrus band 10

    # Create a shadow region in the bottom-left corner
    img[7, 5:, :5] = 500  # NIR band 8 - low in shadows
    img[11, 5:, :5] = 400  # SWIR1 band 11 - low in shadows
    img[1, 5:, :5] = 800  # Blue band 2 - medium-low in shadows

    return img


def test_format_dates_leap_year(leap_test_arrays_dict: np.ndarray) -> None:
    """Test that dates are correctly formatted for leap year."""
    dates = leap_test_arrays_dict["dates"][:, 0]  # use S1 dates
    formatted_dates = _format_dates(dates)

    # Check range (0-365 for leap year)
    assert np.min(formatted_dates) >= 0
    assert np.max(formatted_dates) <= 365

    # Check that dates are sorted
    assert np.all(np.diff(formatted_dates) >= 0)


def test_format_dates_non_leap_year(non_leap_test_arrays_dict: np.ndarray) -> None:
    """Test that dates are correctly formatted for non-leap year."""
    dates = non_leap_test_arrays_dict["dates"][:, 0]  # use S1 dates
    formatted_dates = _format_dates(dates)

    # Check range (0-364 for non-leap year)
    assert np.min(formatted_dates) >= 0
    assert np.max(formatted_dates) <= 364

    # Check that dates are sorted
    assert np.all(np.diff(formatted_dates) >= 0)


def test_pad_missing_dates_leap_year(leap_test_arrays_dict: np.ndarray) -> None:
    """Test padding for leap year based on actual usage pattern."""
    # Get data from the test arrays dict
    data = leap_test_arrays_dict["data"]

    # Format dates for both sensors
    s1_dates = _format_dates(leap_test_arrays_dict["dates"][:, 0])
    s2_dates = _format_dates(leap_test_arrays_dict["dates"][:, 1])

    # Store for verification
    s1_values = data[:, 0:2]
    s2_values = data[:, 2:]

    # Create a 2D tensor for dates (matching real-world usage)
    # First column is S1 dates, second column is S2 dates
    dates_tensor = torch.zeros((max(len(s1_dates), len(s2_dates)), 2), dtype=torch.long)

    # Fill in the dates tensors (padding with 0 if needed)
    dates_tensor[: len(s1_dates), 0] = torch.tensor(s1_dates)
    dates_tensor[: len(s2_dates), 1] = torch.tensor(s2_dates)

    # Create all_dates by flattening
    all_dates = dates_tensor.flatten()
    unique_sorted_dates = torch.unique(all_dates, sorted=True)

    # Pad missing dates
    padding_value = 0.0
    padded_data = _pad_missing_dates(
        cast(torch.Tensor, data),
        dates_tensor,
        unique_sorted_dates,
        num_channels_1=2,
        num_channels_2=13,
        padding_value=padding_value,
    )

    # Check shape
    assert padded_data.shape[0] == len(unique_sorted_dates)  # unique timesteps
    assert padded_data.shape[1] == 15  # channels

    # Check that S1 data is correctly placed
    for i, date in enumerate(s1_dates):
        if date in unique_sorted_dates:
            # Find where this date is in the unique_sorted_dates
            date_idx = (unique_sorted_dates == date).nonzero(as_tuple=True)[0].item()
            assert torch.allclose(padded_data[date_idx, :2], s1_values[i])

    # Check that S2 data is correctly placed
    for i, date in enumerate(s2_dates):
        if date in unique_sorted_dates:
            # Find where this date is in the unique_sorted_dates
            date_idx = (unique_sorted_dates == date).nonzero(as_tuple=True)[0].item()
            assert torch.allclose(padded_data[date_idx, 2:], s2_values[i])

    # Check padding for dates with data from only one sensor
    s1_date_set = set(s1_dates.tolist())
    s2_date_set = set(s2_dates.tolist())

    for date in unique_sorted_dates.tolist():
        date_idx = (unique_sorted_dates == date).nonzero(as_tuple=True)[0].item()

        if date not in s1_date_set:
            assert torch.all(padded_data[date_idx, :2] == padding_value)

        if date not in s2_date_set:
            assert torch.all(padded_data[date_idx, 2:] == padding_value)


def test_bbox_to_polygon() -> None:
    bbox = [10.0, 20.0, 30.0, 40.0]
    expected = [
        [10.0, 20.0],
        [30.0, 20.0],
        [30.0, 40.0],
        [10.0, 40.0],
        [10.0, 20.0],
    ]
    assert _bbox_to_polygon(bbox) == expected


def test_process_SAR_and_rescale(test_s1_img: np.ndarray) -> None:
    processed = process_SAR(test_s1_img, db_min=-50.0, db_max=1.0)
    expected = rescale(
        np.clip(test_s1_img, -50.0, 1.0),
        -50.0,
        1.0,
        multiplier=(1.0 - EPSILON),
        shift=EPSILON,
    )
    np.testing.assert_allclose(processed, expected, atol=1e-6)


def test_process_MS_and_rescale(test_s2_img: np.ndarray) -> None:
    processed = process_MS(test_s2_img, intensity_min=0.0, intensity_max=10000.0)
    expected = rescale(
        np.clip(test_s2_img, 0.0, 10000.0),
        0.0,
        10000.0,
        multiplier=(1.0 - EPSILON),
        shift=EPSILON,
    )
    np.testing.assert_allclose(processed, expected, atol=1e-6)


def test_rescale_no_shift(test_s1_img_rescale: np.ndarray) -> None:
    scaled = rescale(test_s1_img_rescale, old_min=-55, old_max=5, multiplier=1.0, shift=0.0)
    expected = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-6)


def test_normalized_difference() -> None:
    ch1 = np.array([1.0, 2.0, 0.0])
    ch2 = np.array([1.0, 1.0, 0.0])
    expected = np.array([0.0, 0.33333334, 0.0])
    result = normalized_difference(ch1, ch2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_get_cloud_mask(test_cloud_img: np.ndarray) -> None:
    """Test the get_cloud_mask function."""
    # Test with different thresholds
    for threshold in [0.2, 0.5, 0.8]:
        # Get cloud mask with binarization
        binary_mask = get_cloud_mask(test_cloud_img, threshold, binarize=True)

        # Get cloud probability without binarization
        prob_mask = get_cloud_mask(test_cloud_img, threshold, binarize=False)

        # Check shapes
        assert binary_mask.shape == (10, 10)
        assert prob_mask.shape == (10, 10)

        # Check value ranges
        assert np.all((binary_mask == 0) | (binary_mask == 1))
        assert np.all(prob_mask >= 0) and np.all(prob_mask <= 1)

        # Check if binarized mask matches probability threshold
        # Allow small floating point differences
        np.testing.assert_allclose(
            binary_mask,
            (prob_mask >= threshold).astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

        # Check if cloudy region has higher probability than rest of image
        assert np.mean(prob_mask[:5, 5:]) > np.mean(prob_mask[5:, :5])


def test_get_cloud_map(test_cloud_img: np.ndarray) -> None:
    """Test the get_cloud_map function with different detectors."""
    # test cloud_cloudshadow_mask detector
    cloud_shadow_mask = get_cloud_map(test_cloud_img, "cloud_cloudshadow_mask")
    assert cloud_shadow_mask.shape == (10, 10)
    assert np.all((cloud_shadow_mask == 0) | (cloud_shadow_mask == 1))

    # check if cloudy region is detected
    assert np.sum(cloud_shadow_mask[:5, 5:]) > 0

    # test with no detector (should return all ones)
    none_mask = get_cloud_map(test_cloud_img, "")
    assert none_mask.shape == (10, 10)
    assert np.all(none_mask == 1)

    # test with s2cloudless detector (requires instance)
    # mock the S2PixelCloudDetector
    mock_detector = MagicMock()
    mock_detector.get_cloud_masks.return_value = np.ones((1, 10, 10))

    s2cloudless_mask = get_cloud_map(test_cloud_img, "s2cloudless_mask", instance=mock_detector)
    assert s2cloudless_mask.shape == (10, 10)
    assert np.all(s2cloudless_mask == 1)

    # verify the detector was called with correctly formatted data
    mock_detector.get_cloud_masks.assert_called_once()
    call_args = mock_detector.get_cloud_masks.call_args[0][0]
    assert call_args.shape == (1, 10, 10, 13)  # batch, height, width, channels
    assert np.max(call_args) <= 1.0  # values should be normalized to [0,1]


def test_get_shadow_mask(test_shadow_img: np.ndarray) -> None:
    """Test the get_shadow_mask function."""
    # get shadow mask
    shadow_mask = get_shadow_mask(test_shadow_img)

    # check shape and values
    assert shadow_mask.shape == (10, 10)
    assert np.all((shadow_mask == 0) | (shadow_mask == -1))

    # check if shadow is detected in the bottom-left region
    shadow_count_bl = np.sum(shadow_mask[5:, :5] == -1)
    shadow_count_other = np.sum(shadow_mask[:5, 5:] == -1)

    # More shadow pixels should be detected in the bottom-left than elsewhere
    assert shadow_count_bl > shadow_count_other


def test_get_cloud_cloudshadow_mask(test_cloud_shadow_img: np.ndarray) -> None:
    """Test the get_cloud_cloudshadow_mask function."""
    # get combined cloud+shadow mask
    combined_mask = get_cloud_cloudshadow_mask(test_cloud_shadow_img)

    # check shape and values
    assert combined_mask.shape == (10, 10)
    assert np.all((combined_mask == 0) | (combined_mask == 1))

    # check if both cloud and shadow are detected
    pixels_topleft = np.sum(combined_mask[:5, :5])
    pixels_topright = np.sum(combined_mask[:5, 5:])
    pixels_bottomleft = np.sum(combined_mask[5:, :5])
    pixels_bottomright = np.sum(combined_mask[5:, 5:])

    # top-right (cloud) and bottom-left (shadow) should have more detected pixels
    assert pixels_topright > pixels_topleft
    assert pixels_bottomleft > pixels_bottomright
