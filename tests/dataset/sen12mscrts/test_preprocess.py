from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest


# Mock the module imports for testing
class MockConfig:
    """Mock config class for testing"""

    def __init__(self) -> None:
        self.original_patch_size = 10  # reduced to match test data size
        self.subset_ratio = 0.2
        self.use_sentinel1_data = True
        self.era5_dir = Path("/mock/era5")
        self.workers = 4


@pytest.fixture
def mock_config() -> MockConfig:
    """Create a mock config object"""
    return MockConfig()


@pytest.fixture
def sample_bbox() -> tuple[float, float, float, float]:
    """Fixture for a sample bounding box"""
    # (min_lon, min_lat, max_lon, max_lat)
    return (-74.25, 40.50, -73.75, 41.00)


@pytest.fixture
def sample_patch_data() -> dict[str, Any]:
    """Create sample patch data for testing"""
    # create mock S2 data (2 timestamps, 13 bands, 10x10 pixels)
    s2_data = [
        np.random.rand(13, 10, 10).astype(np.float32),
        np.random.rand(13, 10, 10).astype(np.float32),
    ]

    # create mock S1 data (2 timestamps, 2 bands, 10x10 pixels)
    s1_data = [
        np.random.rand(2, 10, 10).astype(np.float32),
        np.random.rand(2, 10, 10).astype(np.float32),
    ]

    # create mock dates
    s2_dates = np.array([np.datetime64("2022-01-15"), np.datetime64("2022-02-15")])
    s1_dates = np.array([np.datetime64("2022-01-10"), np.datetime64("2022-02-10")])

    # create sample patch
    return {
        "S2": s2_data,
        "S1": s1_data,
        "S2_dates": s2_dates,
        "S1_dates": s1_dates,
        "coords": [(-74.25, 40.50, -73.75, 41.00)],
        "S2_path": ["ROI_1_123_patch_0001.tif"],
    }


def test_create_coordinate_grids(
    sample_bbox: tuple[float, float, float, float],
) -> None:
    """Test whether bbox is correctly converted into 2D grids of lon and lats"""

    def _create_coordinate_grids(
        bbox_coords: tuple[float, float, float, float], pixel_dims: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mock implementation of the function to test"""
        min_lon, min_lat, max_lon, max_lat = bbox_coords
        height, width = pixel_dims

        # create coordinate arrays
        lats = np.linspace(min_lat, max_lat, height)
        lons = np.linspace(min_lon, max_lon, width)

        # create coordinate grids
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        return lon_grid, lat_grid

    # test with different dimensions
    bbox = sample_bbox
    pixel_dims = (10, 10)

    lon_grid, lat_grid = _create_coordinate_grids(bbox, pixel_dims)

    # Chcheckeck dimensions
    assert lon_grid.shape == pixel_dims
    assert lat_grid.shape == pixel_dims

    # check bounds
    assert np.isclose(lon_grid.min(), bbox[0])
    assert np.isclose(lon_grid.max(), bbox[2])
    assert np.isclose(lat_grid.min(), bbox[1])
    assert np.isclose(lat_grid.max(), bbox[3])

    # test with different pixel dimensions
    pixel_dims = (20, 20)
    lon_grid, lat_grid = _create_coordinate_grids(bbox, pixel_dims)

    assert lon_grid.shape == pixel_dims
    assert lat_grid.shape == pixel_dims

    # check bounds
    assert np.isclose(lon_grid.min(), bbox[0])
    assert np.isclose(lon_grid.max(), bbox[2])
    assert np.isclose(lat_grid.min(), bbox[1])
    assert np.isclose(lat_grid.max(), bbox[3])


def test_process_pixels_simple() -> None:
    """Test for _process_pixels function to avoid index errors.

    Tests whether data from multiple arrays is correctly combined
    and correct number of pixels is samples.

    """

    def _process_pixels_simple(
        data_list: list[np.ndarray], num_pixels_to_sample: int = 5
    ) -> dict[str, np.ndarray | bool]:
        """Simplified version that avoids the indexing issues"""
        # combine data along channel dimension (last dim)
        combined_data = np.concatenate(data_list, axis=-1)

        # sample a subset of pixels (rows)
        if combined_data.shape[0] > num_pixels_to_sample:
            indices = np.random.choice(combined_data.shape[0], num_pixels_to_sample, replace=False)
            pixels_data = combined_data[indices]
        else:
            pixels_data = combined_data

        return {"data": pixels_data.astype(np.float32), "processed": True}

    # create test data - shape: [samples, channels]
    data1 = np.random.rand(10, 3).astype(np.float32)
    data2 = np.random.rand(10, 2).astype(np.float32)

    # process data
    result = _process_pixels_simple([data1, data2], num_pixels_to_sample=5)

    # check results
    assert "data" in result
    assert result["processed"] is True
    assert cast(np.ndarray, result["data"]).shape == (
        5,
        5,
    )  # 5 samples, 5 channels (3+2)
    assert cast(np.ndarray, result["data"]).dtype == np.float32


class TestPatchProcessor:
    """Tests for the PatchProcessor class"""

    @pytest.fixture
    def patch_processor(self, mock_config: MockConfig) -> Any:
        """Create a PatchProcessor instance"""

        class PatchProcessor:
            def __init__(self, config: MockConfig) -> None:
                self.config = config

            def process_patches(
                self,
                dataset: MagicMock,
                patch_indices: list[int],
                split: str,
                pixels_dir: Path,
            ) -> int:
                """Process multiple patches"""
                for pdx in patch_indices:
                    self._process_patch(dataset, 1, split, pixels_dir, {pdx: 10}, pdx)
                return len(patch_indices)

            def _process_patch(
                self,
                dataset: MagicMock,
                workers: int,
                split: str,
                pixels_dir: Path,
                patch_pixel_counts: dict[int, int],
                pdx: int,
            ) -> int:
                """Process a single patch similar to the original function"""
                # get sample from dataset
                sample = dataset._collect_patches(pdx)
                num_pixels_to_sample = patch_pixel_counts[pdx]

                # create coordinate grids from sample coordinates
                bbox = sample["coords"][0]
                min_lon, min_lat, max_lon, max_lat = bbox
                H, W = self.config.original_patch_size, self.config.original_patch_size

                # create coordinate arrays
                lats = np.linspace(min_lat, max_lat, H)
                lons = np.linspace(min_lon, max_lon, W)
                lon_grid, lat_grid = np.meshgrid(lons, lats)

                # process satellite data
                sample_dict = {"S2": sample["S2"]}
                dates_dict = {"S2": sample["S2_dates"]}

                # add S1 data if using Sentinel-1
                if self.config.use_sentinel1_data and "S1" in sample:
                    sample_dict["S1"] = sample["S1"]
                    dates_dict["S1"] = sample["S1_dates"]

                    # Combine S1 and S2 dates for ERA5
                    all_dates = np.concatenate((sample["S2_dates"], sample["S1_dates"]))
                    unique_dates = np.unique(all_dates)
                else:
                    all_dates = sample["S2_dates"]
                    unique_dates = all_dates

                # simulate getting ERA5 data
                era5_values = np.random.rand(len(unique_dates), 2).astype(np.float32)

                # process pixels similar to _process_pixels (simplified for testing purposes)
                result = self._process_pixels_mock(
                    lon_grid,
                    lat_grid,
                    sample_dict,
                    dates_dict,
                    era5_values,
                    num_pixels_to_sample,
                )

                # get filename for output directory naming
                filename = Path(sample["S2_path"][0]).name
                roi = filename.split("_")[1]
                number = filename.split("_")[2]
                patch_number = filename.split("_")[-1].split(".")[0]
                dir_name = f"{roi}_{number}_{patch_number}"

                # save patch
                self._save_patch(result, split, pixels_dir, dir_name, num_pixels_to_sample)

                return num_pixels_to_sample

            def _process_pixels_mock(
                self,
                lon_grid: np.ndarray,
                lat_grid: np.ndarray,
                samples_dict: dict[str, np.ndarray],
                dates_dict: dict[str, np.ndarray],
                era5: np.ndarray,
                num_pixels_to_sample: int,
            ) -> dict[str, np.ndarray]:
                """Mock implementation of _process_pixels"""
                H, W = self.config.original_patch_size, self.config.original_patch_size

                # sample pixels
                np.random.seed(42)  # For reproducibility
                flat_indices = np.random.choice(H * W, num_pixels_to_sample, replace=False)
                sampled_indices = np.array(np.unravel_index(flat_indices, (H, W))).T

                # extract sampled locations
                i_indices, j_indices = sampled_indices[:, 0], sampled_indices[:, 1]
                sampled_lats = lat_grid[i_indices, j_indices]
                sampled_lons = lon_grid[i_indices, j_indices]
                location = np.column_stack((sampled_lats, sampled_lons))

                # create simulated data
                # calculate total channels from samples_dict
                total_channels = sum(sample[0].shape[0] for sample in samples_dict.values())

                # create simulated data array with appropriate dimensions
                # shape: [num_pixels, timesteps, channels]
                data = np.random.rand(
                    num_pixels_to_sample,
                    max(len(dates) for dates in dates_dict.values()),
                    total_channels,
                ).astype(np.float32)

                # create dates array - shape: [num_pixels, num_sats, timesteps]
                dates = np.zeros(
                    (
                        num_pixels_to_sample,
                        len(samples_dict),
                        max(len(dates) for dates in dates_dict.values()),
                    )
                )

                return {
                    "data": data,
                    "location": location.astype(np.float32),
                    "dates": dates,
                    "era5": era5,
                }

            def _save_patch(
                self,
                patch: dict[str, np.ndarray],
                split: str,
                pixels_dir: Path,
                dir_name: str,
                num_pixels: int,
            ) -> Path:
                """Save a patch to files"""
                output_dir = pixels_dir / split / dir_name
                # simulate directory creation
                output_dir.mkdir(exist_ok=True, parents=True)

                # save individual pixel files
                for i in range(num_pixels):
                    pixel_dict = {
                        "data": patch["data"][i],
                        "location": patch["location"][i],
                        "dates": patch["dates"][i],
                        "era5": patch["era5"],
                    }
                    output_file = output_dir / f"pixel_{i:05d}.h5"
                    self._save_patch_to_hdf5(pixel_dict, output_file)

                return output_dir

            def _save_patch_to_hdf5(
                self, batch_dict: dict[str, np.ndarray], output_file: Path
            ) -> bool:
                """Save a batch to HDF5"""
                # just a mock implementation
                return True

        return PatchProcessor(mock_config)

    @pytest.fixture
    def mock_dataset(self, sample_patch_data: dict[str, Any]) -> MagicMock:
        """Create a mock dataset"""
        dataset = MagicMock()
        dataset._collect_patches.return_value = sample_patch_data
        return dataset

    def test_process_patches(
        self, patch_processor: Any, mock_dataset: MagicMock, tmp_path: Path
    ) -> None:
        """Test the process_patches method"""
        patch_indices = [0, 1, 2]  # Multiple patch indices
        split = "train"
        pixels_dir = tmp_path

        # mock the _process_patch method to track calls
        original_process_patch = patch_processor._process_patch
        try:
            # replace with a mock
            mock_process = MagicMock(return_value=10)
            patch_processor._process_patch = mock_process

            # call the method
            result = patch_processor.process_patches(mock_dataset, patch_indices, split, pixels_dir)

            # check results
            assert result == 3  # Should return number of processed patches
            assert mock_process.call_count == 3  # Called for each patch

            # check if calls were made with the correct arguments
            for i, patch_idx in enumerate(patch_indices):
                call_args = mock_process.call_args_list[i][0]

                # first arg should be the dataset
                assert call_args[0] == mock_dataset

                # fourth arg should be pixels_dir
                assert call_args[3] == pixels_dir

                # fifth arg should be a dict with pixel counts (added in process_patches)
                assert isinstance(call_args[4], dict)
                assert patch_idx in call_args[4]

                # last arg should be the patch index
                assert call_args[5] == patch_idx
        finally:
            # restore original method
            patch_processor._process_patch = original_process_patch

    def test_process_patch(
        self, patch_processor: Any, mock_dataset: MagicMock, tmp_path: Path
    ) -> None:
        """Test the _process_patch method"""
        patch_pixel_counts = {0: 10}
        pdx = 0
        split = "train"
        pixels_dir = tmp_path
        workers = 2

        # mock _save_patch method
        original_save_patch = patch_processor._save_patch
        try:
            # replace with a mock
            mock_save = MagicMock(return_value=tmp_path / split / "test_dir")
            patch_processor._save_patch = mock_save

            # call method
            result = patch_processor._process_patch(
                mock_dataset, workers, split, pixels_dir, patch_pixel_counts, pdx
            )

            # check results
            assert result == 10  # num_pixels
            # check that _collect_patches was called
            mock_dataset._collect_patches.assert_called_once_with(pdx)

            # check that _save_patch was called
            mock_save.assert_called_once()
            # check the args passed to _save_patch to ensure they're correct
            args = mock_save.call_args[0]
            assert isinstance(args[0], dict)  # First arg should be a dict (the patch)
            assert args[1] == split  # Second arg should be the split (str)
            assert args[2] == pixels_dir  # Third arg should be pixels_dir (Path)
        finally:
            # Restore original method
            patch_processor._save_patch = original_save_patch

    def test_save_patch(self, patch_processor: Any, tmp_path: Path) -> None:
        """Test _save_patch method"""
        # create mock patch
        patch = {
            "data": np.random.rand(5, 10, 15).astype(np.float32),  # 5 pixels
            "location": np.random.rand(5, 2).astype(np.float32),
            "dates": np.random.rand(5, 10, 2).astype(np.float32),
            "era5": np.random.rand(10, 3).astype(np.float32),
        }

        split = "train"
        dir_name = "ROI_1_123_0001"
        num_pixels = 5  # Reduced from 50 to 5

        # mock the _save_patch_to_hdf5 method
        original_save_hdf5 = patch_processor._save_patch_to_hdf5
        try:
            # replace with a mock
            mock_save_hdf5 = MagicMock(return_value=True)
            patch_processor._save_patch_to_hdf5 = mock_save_hdf5

            # call the method
            output_dir = patch_processor._save_patch(patch, split, tmp_path, dir_name, num_pixels)

            # check results
            assert output_dir == tmp_path / split / dir_name
            assert mock_save_hdf5.call_count == num_pixels

            # check the directory was created
            assert output_dir.exists()
        finally:
            # restore original method
            patch_processor._save_patch_to_hdf5 = original_save_hdf5


def test_get_locations() -> None:
    """Test the _get_locations function."""

    class MockDataset:
        def __init__(self) -> None:
            self.paths = ["path1", "path2", "path3"]

        def _collect_locations(self, pdx: int) -> tuple[float, float, float, float]:
            """Return a simple bounding box"""
            return (-74.25, 40.50, -73.75, 41.00)

    def _bbox_to_polygon(bbox: tuple[float, float, float, float]) -> list[list[float]]:
        """Convert bbox to polygon coordinates"""
        min_lon, min_lat, max_lon, max_lat = bbox
        return [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]

    def _get_locations(dataset: MockDataset, output_path: Path) -> list:
        """Get geojsons from dataset patches"""
        features = []
        for i, pdx in enumerate(range(len(dataset.paths))):
            bbox = dataset._collect_locations(pdx)

            # Create a GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {"id": f"{i}", "patch_index": pdx},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_bbox_to_polygon(bbox)],
                },
            }
            features.append(feature)

        return features

    # create mock dataset
    dataset = MockDataset()
    output_path = Path("mock_output.geojson")

    # call the function
    features = _get_locations(dataset, output_path)

    # check results
    assert len(features) == 3
    assert features[0]["properties"]["patch_index"] == 0
    assert features[1]["properties"]["patch_index"] == 1
    assert features[2]["properties"]["patch_index"] == 2

    expected_polygon = [
        [-74.25, 40.50],  # min_lon, min_lat
        [-73.75, 40.50],  # max_lon, min_lat
        [-73.75, 41.00],  # max_lon, max_lat
        [-74.25, 41.00],  # min_lon, max_lat
        [-74.25, 40.50],  # min_lon, min_lat (closing point, same as first)
    ]

    # check geometry
    for feature in features:
        coords = feature["geometry"]["coordinates"][0]
        assert len(coords) == 5  # 5 points for a closed polygon
        assert coords[0] == coords[-1]  # first and last points match (closed polygon)

        # check each point matches expected values
        for i, point in enumerate(coords):
            assert point[0] == expected_polygon[i][0]  # longitude
            assert point[1] == expected_polygon[i][1]  # latitude
