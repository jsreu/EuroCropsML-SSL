from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import rasterio
import scipy
import scipy.signal as scisig
import torch
from rasterio.io import DatasetReader

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector
from scipy.ndimage import gaussian_filter

from eurocropsssl.settings import EPSILON


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
    """Pad missing data with a certain value between two groups of channels."""

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


def _bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    """Convert a bounding box [min_lon, min_lat, max_lon, max_lat] to a GeoJSON polygon."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # Define coordinates for a polygon
    coordinates = [
        [min_lon, min_lat],  # bottom-left
        [max_lon, min_lat],  # bottom-right
        [max_lon, max_lat],  # top-right
        [min_lon, max_lat],  # top-left
        [min_lon, min_lat],  # bottom-left
    ]
    return coordinates


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_img: Path) -> DatasetReader:
    """Load GeoTIFF from file."""
    tif = rasterio.open(path_img)
    return tif


def read_img(tif: DatasetReader) -> np.ndarray:
    """Read GeoTIFF image data into numpy."""
    data: np.ndarray = tif.read().astype(np.float32)
    return data


def rescale(
    img: np.ndarray,
    old_min: float,
    old_max: float,
    multiplier: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """Feature scaling."""
    old_range = old_max - old_min
    img = (img - old_min) / old_range
    return img * multiplier + shift


def process_MS(
    img: np.ndarray, intensity_min: float = 0.0, intensity_max: float = 10000.0
) -> np.ndarray:
    """Function for scaling Sentinel-2 multi-spectral data to (0,1]."""
    # intensity clipping to a global unified MS intensity range
    img = np.clip(img, intensity_min, intensity_max)
    # project to (0,1], preserve global intensities (across patches)
    img = rescale(img, intensity_min, intensity_max, multiplier=(1 - EPSILON), shift=EPSILON)
    return img


def process_SAR(img: np.ndarray, db_min: float = -50.0, db_max: float = 1.0) -> np.ndarray:
    """Function for scaling Sentinel-1 radar data to (0,1]."""
    # intensity clipping to a global unified SAR dB range
    img = np.clip(img, db_min, db_max)
    # project to (0,1], preserve global intensities (across patches)
    img = rescale(
        img,
        db_min,
        db_max,
        multiplier=(1 - EPSILON),
        shift=EPSILON,
    )
    return img


def get_cloud_cloudshadow_mask(img: np.ndarray, cloud_threshold: float = 0.2) -> np.ndarray:
    """Masking clouds and cloud shadows from optical data.

    img: [13 x H x W] S2 image (of arbitrary resolution H,W)
        They ar expected to have their default ranges and not be value-standardized yet.
    cloud_threshold: Threshold to detct clouds/shadows. The higher, the more conservative the masks
     (i.e. less pixels labeled clouds/shadows)

    Returns:
        Cloud & shadow segmentation mask (of same resolution)
    """
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = 1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    return cloud_cloudshadow_mask


def get_cloud_map(
    img: np.ndarray,
    detector: str,
    instance: S2PixelCloudDetector | None = None,
    s2_intensity_min: float = 0.0,
    s2_intensity_max: float = 10000.0,
) -> np.ndarray:
    """Obtaining (shadow) cloud mask for later on masking out clouds from optical data."""
    if detector == "cloud_cloudshadow_mask":
        threshold = 0.2  # set to e.g. 0.2 or 0.4
        mask = get_cloud_cloudshadow_mask(
            np.clip(img, s2_intensity_min, s2_intensity_max), threshold
        )
    elif detector == "s2cloudless_map":
        threshold = 0.5
        instance = cast(S2PixelCloudDetector, instance)
        mask = instance.get_cloud_probability_maps(
            np.moveaxis(
                np.clip(img, s2_intensity_min, s2_intensity_max) / s2_intensity_max,
                0,
                -1,
            )[None, ...]
        )[0, ...]
        mask[mask < threshold] = 0
        mask = gaussian_filter(mask, sigma=2).astype(np.float32)
    elif detector == "s2cloudless_mask":
        instance = cast(S2PixelCloudDetector, instance)
        mask = instance.get_cloud_masks(
            np.moveaxis(
                np.clip(img, s2_intensity_min, s2_intensity_max) / s2_intensity_max,
                0,
                -1,
            )[None, ...]
        )[0, ...]
    else:
        mask = np.ones((img.shape[-1], img.shape[-1]))
    return mask


# obtained from https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/util/detect_cloudshadow.py


def normalized_difference(channel1: np.ndarray, channel2: np.ndarray) -> np.ndarray:
    """Calculate normalized difference between two channels ((ch1-ch2)/(ch1+ch2)).

    For example used for calculating indices like NDVI, NDWI, NDSI, etc.

    Args:
        channel1: First spectral band array
        channel2: Second spectral band array

    Returns:
        Normalized difference array.
    """
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = EPSILON  # checking for 0 divisions
    result: np.ndarray = subchan / sumchan
    return result


def get_shadow_mask(data_image: np.ndarray, s2_intensity_max: float = 10000.0) -> np.ndarray:
    """Generate cloud shadow mask for Sentinel-2 imagery using spectral indices.

    Uses Blue (B2), NIR (B8), and SWIR1 (B11) bands to detect shadows based on:
    - Cloud-Shadow Index (CSI): average of NIR and SWIR1
    - Blue band threshold for water body detection
    Applies median filtering to reduce noise.

    Args:
        data_image: Multi-band Sentinel-2 image array of shape (channels, height, width)
           with raw DN values (0-s2_intensity_max).
        s2_intensity_max: Max intensity value for S2.
            Defaults to 10000.0.

    Returns:
        Binary shadow mask where -1 indicates shadow and 0 indicates no shadow.
    """
    data_image = data_image / s2_intensity_max
    (ch, r, c) = data_image.shape
    shadowmask: np.ndarray = np.zeros((r, c), dtype=np.float32)

    BB = data_image[1]
    BNIR = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.0

    t3 = 3 / 4  # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return shadowmask


def get_cloud_mask(
    data_image: np.ndarray,
    cloud_threshold: float,
    binarize: bool = False,
    s2_intensity_max: float = 10000.0,
    use_moist_check: bool = False,
) -> np.ndarray:
    """Generate cloud mask for Sentinel-2 imagery using multiple spectral checks.

    Implements a series of spectral tests to identify clouds:
    1. Brightness in blue and aerosol/cirrus bands
    2. Optional moisture check using NDMI (Normalized Difference Moisture Index)
    3. Snow discrimination using NDSI (Normalized Difference Snow Index)
    Includes morphological operations and spatial smoothing.

    Args:
        data_image: Multi-band Sentinel-2 image array of shape (channels, height, width)
            with raw DN values (0-10000)
        cloud_threshold: Threshold for cloud probability (typically 0.2-0.4)
        binarize: If True, returns binary mask. If False, returns probability. Defaults to False.
        s2_intensity_max: Max intensity value for S2. Defaults to 10000.0.
        use_moist_check: If True, includes moisture index in cloud detection. Defaults to False.

    Returns:
        Cloud mask/probability array. If binarized, values are 0 (no cloud) or 1 (cloud).
            Otherwise, values range from 0-1 indicating cloud probability.

    Reference:
        Adapted from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/\
cloud-masking-sentinel2.ipynb
    """

    data_image = data_image / s2_intensity_max
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c), dtype=np.float32)
    # 1. Brightness in blue and aerosol/cirrus bands
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(data_image[1], old_min=0.1, old_max=0.5))
    score = np.minimum(score, rescale(data_image[0], old_min=0.1, old_max=0.3))
    score = np.minimum(score, rescale((data_image[0] + data_image[10]), old_min=0.4, old_max=0.9))
    score = np.minimum(
        score,
        rescale((data_image[3] + data_image[2] + data_image[1]), old_min=0.2, old_max=0.8),
    )
    # 2. Optional moisture check using NDMI
    if use_moist_check:
        # Clouds are moist
        ndmi = normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, rescale(ndmi, old_min=-0.1, old_max=0.1))

    # 3. Snow discrimination using NDSI (Normalized Difference Snow Index)
    ndsi = normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, rescale(ndsi, old_min=0.8, old_max=0.6))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize**2)

    score = scipy.ndimage.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode="same")

    score = cast(np.ndarray, np.clip(score, 0.00001, 1.0))

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold] = 0

    return score
