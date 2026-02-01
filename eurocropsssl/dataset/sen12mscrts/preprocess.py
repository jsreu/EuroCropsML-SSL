import json
import logging
import multiprocessing as mp_orig
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from eurocropsssl.dataset.era5.preprocess import get_era5_values
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.dataset import SEN12MSCRTS
from eurocropsssl.dataset.sen12mscrts.utils import _bbox_to_polygon, _format_dates
from eurocropsssl.settings import EPSILON, Settings
from eurocropsssl.train.utils import set_seed

logger = logging.getLogger(__name__)


def _create_coordinate_grids(
    bbox_coords: tuple[float, float, float, float], pixel_dims: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Create 2D grids of longitudes and latitudes for all pixels within a 2D patch.

    Arguments::
        bbox_coords: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
        pixel_dims: Dimensions of the pixel grid (width, height)

    Returns:
        Tuple with longitude and latitude grid.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    height, width = pixel_dims

    # Create coordinate arrays
    lats = np.linspace(min_lat, max_lat, height)
    lons = np.linspace(min_lon, max_lon, width)

    # Create coordinate grids
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    return lon_grid, lat_grid


def _process_pixels(
    config: SEN12MSCRTSDatasetConfig,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    samples_dict: dict[
        str, list[np.ndarray]
    ],  # keys: S2 (and S1 if self.config.use_sentinel1_data is True)
    dates_dict: dict[str, np.ndarray],
    era5: np.ndarray,
    num_pixels_to_sample: int,
) -> dict[str, np.ndarray]:

    H, W = config.original_patch_size, config.original_patch_size
    num_pixels = H * W

    data_list = []
    for _, samples in samples_dict.items():
        stacked = np.stack(samples, axis=0)  # [T, C, H, W]
        data = np.transpose(stacked, (3, 2, 0, 1))  # [H, W, T, C]
        data_list.append(data)

    # Combine satellite data along channel dimension
    combined_data = np.concatenate(data_list, axis=3)

    if num_pixels_to_sample < num_pixels:
        # randomly sample flat indices
        flat_indices = np.random.choice(num_pixels, num_pixels_to_sample, replace=False)
        # convert to 2D positions
        sampled_indices = np.array(np.unravel_index(flat_indices, (H, W))).T

        # extract sampled pixels
        i_indices, j_indices = sampled_indices[:, 0], sampled_indices[:, 1]

        # index original patch with sampled indices
        pixels_data = combined_data[i_indices, j_indices]
        sampled_lats = lat_grid[i_indices, j_indices]
        sampled_lons = lon_grid[i_indices, j_indices]

        location = np.column_stack((sampled_lats, sampled_lons))
        num_pixels = num_pixels_to_sample
    else:
        # reshape
        pixels_data = combined_data.reshape(num_pixels, -1, combined_data.shape[-1])
        lat_coords = lat_grid.ravel()
        lon_coords = lon_grid.ravel()
        location = np.column_stack((lat_coords, lon_coords))

    # Format dates
    dates_array = np.stack([_format_dates(dates_dict[sat_type]) for sat_type in samples_dict])
    # repeat for the number of pixels
    dates = np.broadcast_to(
        dates_array[np.newaxis, :, :],
        (num_pixels, dates_array.shape[0], dates_array.shape[1]),
    )

    return {
        "data": pixels_data.astype(np.float32),  # Shape: [num_pixels, timesteps, channels]
        "location": location.astype(np.float32),  # Shape: [num_pixels, 2] (lat, lon)
        "dates": dates,  # Shape: [num_pixels, num_sats, timesteps]
        "era5": era5.astype(np.float32),  # Shape: [unique_timesteps, era5_variables]
    }


class PatchProcessor:
    """Processor for extracting and saving pixel-level data from SEN12MS-CR-TS patches.

    Handles sampling pixels from satellite data patches, extracting corresponding
    spatiotemporal information, and saving the results to HDF5 files. Supports
    configurable sampling rates, multi-sensor data fusion, and parallel processing.
    """

    def __init__(self, config: SEN12MSCRTSDatasetConfig):
        self.config = config

    def process_patches(
        self,
        dataset: SEN12MSCRTS,
        patch_indices: list[int],
        split: Literal["train", "val"],
        pixels_dir: Path,
    ) -> None:
        """Process individual patches from SEN12MSCRTS dataset.

        Args:
            dataset: SEN12MSCRTS dataset instance
            patch_indices: List of patch indices to process.
            split: Train or validation split. Only used for directory naming.
            pixels_dir: Directory to save patches in.

        """
        # calculate number of pixels to sample from each patch
        subset_ratio = max(0.0001, min(1.0, self.config.subset_ratio))  # setting minimum to 0.0001
        if subset_ratio != self.config.subset_ratio:
            logger.info(
                f"{self.config.subset_ratio} is not a valid value. "
                f"Setting subset_ratio to {subset_ratio}."
            )
        logger.info(f"Sampling {subset_ratio*100:.1f}% of the full SEN12MS-CR-TS dataset.")
        total_patches = len(patch_indices)
        num_pixels = self.config.original_patch_size**2
        if self.config.subset_ratio < 1.0:
            base_pixels_per_patch = int(self.config.subset_ratio * num_pixels)
            total_pixels_subset: int = round(total_patches * self.config.subset_ratio * num_pixels)
            remainder = total_pixels_subset - (base_pixels_per_patch * total_patches)
            # sample random patch indices to allocate the remaining pixels
            remainder_indices = np.random.choice(total_patches, remainder, replace=False)
            pixel_allocations: np.ndarray = np.full(total_patches, base_pixels_per_patch)
            pixel_allocations[remainder_indices] += 1
            # convert to dict mapping patch index and number of pixels
            patch_pixel_counts: dict[int, int] = {
                patch_idx: pixel_allocations[i] for i, patch_idx in enumerate(patch_indices)
            }
        else:
            patch_pixel_counts = dict.fromkeys(patch_indices, num_pixels)

        max_workers = min(mp_orig.cpu_count(), max(1, min(len(patch_indices), self.config.workers)))
        workers_era5 = int(mp_orig.cpu_count() / max_workers)
        func = partial(
            self._process_patch,
            dataset,
            workers_era5,
            split,
            pixels_dir,
            patch_pixel_counts,
        )

        with mp_orig.Pool(processes=max_workers) as p:
            te = tqdm(
                total=len(patch_indices),
                desc=f"Processing SEN12MS-CR-TS {split} patches",
                position=0,
            )
            te.refresh()  # display progress bar immediately

            process_iter = p.imap_unordered(func, patch_indices)
            for _ in process_iter:
                te.update(n=1)

    def _process_patch(
        self,
        dataset: SEN12MSCRTS,
        workers: int,
        split: Literal["train", "val"],
        pixels_dir: Path,
        patch_pixel_counts: dict[int, int],
        pdx: int,
    ) -> None:
        num_pixels_to_sample = patch_pixel_counts[pdx]
        sample = dataset._collect_patches(pdx)

        if self.config.sample_type == "cloudy_cloudfree":
            sample = cast(dict, sample["input"])

        lon_grid, lat_grid = _create_coordinate_grids(
            cast(list, sample["coords"])[0],
            (
                int(self.config.original_patch_size),
                int(self.config.original_patch_size),
            ),
        )
        sample_dict = {"S2": cast(list, sample["S2"])}
        dates_dict = {"S2": cast(np.ndarray, sample["S2_dates"])}

        dates: list | np.ndarray = (
            sample["S2_dates"] if isinstance(sample["S2_dates"], (list, np.ndarray)) else []
        )

        # Process S1 dates if needed
        if self.config.use_sentinel1_data:
            sample_dict["S1"] = cast(list, sample["S1"])
            dates_dict["S1"] = cast(np.ndarray, sample["S1_dates"])
            s1_dates = (
                sample["S1_dates"] if isinstance(sample["S1_dates"], (list, np.ndarray)) else []
            )
            dates = np.concatenate((dates, s1_dates), axis=0)
            # get unique dates to query ERA5 data
            dates = np.unique(dates)

        # Use threads for I/O-bound ERA5 processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            coords = cast(list, sample["coords"])[0]

            # we query the ERA5 data for the center point of the patch
            # SEN12MS-CR-TS has resolution 10m per pixel, whereas ERA5 has 30km
            # so this is a reasonable tradeoff between computational costs and accuracy
            era5_values: np.ndarray = np.array(
                list(
                    executor.map(
                        lambda date: get_era5_values(
                            date,
                            coords,
                            self.config.era5_dir,
                            self.config.era5_normalize,
                        ),
                        dates,
                    )
                )
            )

        result = _process_pixels(
            self.config,
            lon_grid,
            lat_grid,
            dict(sorted(sample_dict.items())),  # sorting for S1 being first
            dict(sorted(dates_dict.items())),  # sorting for S1 being first
            era5_values,
            num_pixels_to_sample,
        )

        filename = Path(sample["S2_path"][0]).name  # type: ignore[index]

        roi = filename.split("_")[1]
        number = filename.split("_")[2]
        patch_number = filename.split("_")[-1].split(".")[0]

        dir_name = f"{roi}_{number}_{patch_number}"

        self._save_patch(result, split, pixels_dir, dir_name, num_pixels_to_sample)
        result.clear()

    def _save_patch(
        self,
        patch: dict[str, np.ndarray],
        split: Literal["train", "val"],
        pixels_dir: Path,
        dir_name: str,
        num_pixels_to_sample: int,
    ) -> None:

        output_dir = pixels_dir / split / dir_name
        output_dir.mkdir(exist_ok=True, parents=True)

        patch["dates"] = np.transpose(patch["dates"], (0, 2, 1))

        for i in range(num_pixels_to_sample):
            # Create pixel data
            pixel_dict: dict[str, np.ndarray] = {
                "data": patch["data"][i],  # Shape: [timesteps, channels]
                "location": patch["location"][i],  # Shape: [2] (lat, lon)
                "dates": patch["dates"][
                    i
                ].squeeze(),  # Shape: [timesteps, num_sats] ([timesteps] if num_sats==1)
                "era5": patch["era5"],  # Shape: [unique_timesteps, era5_variables]
            }

            # Save patch to HDF5
            output_file = output_dir / f"pixel_{i:05d}.h5"
            self._save_patch_to_hdf5(pixel_dict, output_file)

    def _save_patch_to_hdf5(self, batch_dict: dict[str, np.ndarray], output_file: Path) -> None:
        """Save a single batch to HDF5 format.

        Args:
            batch_dict: Dictionary containing batch data
            output_file: Path to save the HDF5 file
        """
        with h5py.File(output_file, "w") as f:
            # Create datasets for each array
            for key, data in batch_dict.items():
                f.create_dataset(key, data=data)


def _get_locations(dataset: SEN12MSCRTS, output_path: Path) -> gpd.GeoDataFrame:
    """Get polygons from SEN12MS-CR-TS patches."""

    # process patches
    patch_indices = range(0, len(dataset.paths))
    features = []
    for i, pdx in tqdm(
        enumerate(patch_indices),
        desc="Collecting polygons from SEN12MS-CR-TS...",
        total=len(patch_indices),
    ):
        bbox = dataset._collect_locations(pdx)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {"id": f"{i}", "patch_index": pdx},
            "geometry": {"type": "Polygon", "coordinates": [_bbox_to_polygon(bbox)]},
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    geojson_str = json.dumps(geojson)

    with open(output_path, "w") as f:
        f.write(geojson_str)

    return gpd.GeoDataFrame.from_features(geojson["features"])


def _filter_polygons(gdf: gpd.GeoDataFrame, output_path: Path) -> list[int]:
    """Find and remove overlapping polygons."""
    logger.info("Removing duplicate polygons...")
    to_drop = set()
    # Create spatial index for efficiency
    spatial_index = gdf.sindex

    for idx, row in gdf.iterrows():
        if idx in to_drop:
            continue

        geometry = row.geometry
        error_margin = EPSILON * geometry.area  # error margin to find actual overlapping polygons

        # Find potential matches
        possible_matches_idx = list(spatial_index.intersection(geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_idx]

        # Only look at indices higher than current to avoid checking pairs twice
        possible_matches = possible_matches[possible_matches.index > idx]

        for match_idx, match_row in possible_matches.iterrows():
            if match_row.geometry.intersects(geometry):
                intersection = geometry.intersection(match_row.geometry)
                if intersection.area > error_margin:
                    # Keep the first polygon, drop the second one
                    to_drop.add(match_idx)

    # Drop intersecting polygons
    gdf_no_intersects = gdf.drop(list(to_drop))

    logger.info(f"Found {len(to_drop)} overlapping polygons.")

    indices_to_keep = list(gdf_no_intersects.patch_index)

    # Save the IDs as numpy arrays
    np.save(output_path.joinpath("sen12mscrts_indices.npy"), np.array(indices_to_keep))
    logger.info(f"Saved {len(indices_to_keep)} non-overlapping features to NumPy.")

    # Save overlapping and non-overlapping features to geojson for potential further analysis.
    gdf_no_intersects.to_file(
        output_path.joinpath("sen12mscrts_nonoverlap.geojson"), driver="GeoJSON"
    )
    logger.info(f"Saved {len(gdf_no_intersects)} non-overlapping features to GeoJSON.")

    gdf_dropped = gdf.loc[list(to_drop)].copy()
    gdf_dropped.to_file(output_path.joinpath("sen12mscrts_overlap.geojson"), driver="GeoJSON")
    logger.info(f"Saved {len(gdf_dropped)} overlapping features to GeoJSON.")

    return indices_to_keep


def generate_pixel_timeseries(config: SEN12MSCRTSDatasetConfig) -> None:
    """Generate pixel time series from Sentinel patch."""
    pixels_dir: Path = config.raw_data_dir.parent.joinpath("PixelSEN12MSCRTS")
    pixels_dir.mkdir(exist_ok=True, parents=True)
    meta_data_dir: Path = pixels_dir.joinpath("meta_data")
    meta_data_dir.mkdir(exist_ok=True, parents=True)

    set_seed(Settings().seed)

    dataset: SEN12MSCRTS = SEN12MSCRTS(
        root_dir=config.raw_data_dir.joinpath("SEN12MSCRTS"),
        split=config.splits,
        region=config.region,
        sample_type=config.sample_type,
    )

    indices_file = meta_data_dir.joinpath("sen12mscrts_indices.npy")
    if indices_file.exists():
        logger.info("Loading filtered non-overlapping patch indices.")
        dataset_indices = np.load(indices_file).tolist()
    else:
        gdf = _get_locations(dataset, meta_data_dir.joinpath("locations.geojson"))
        dataset_indices = _filter_polygons(gdf, meta_data_dir)

    train_idxs, val_idxs = train_test_split(
        dataset_indices, test_size=0.2, random_state=Settings().seed
    )

    patch_processor = PatchProcessor(config)
    # process train patches
    patch_processor.process_patches(dataset, train_idxs, "train", pixels_dir)
    # process validation patches
    patch_processor.process_patches(dataset, val_idxs, "val", pixels_dir)
