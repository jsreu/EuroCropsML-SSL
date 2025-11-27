# obtained from https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py
# format adjusted s.t. it fits the EUROCROPSSSL pipeline and purpose
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from natsort import natsorted

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector
from tqdm import tqdm

from eurocropsssl.dataset.sen12mscrts.utils import (
    get_cloud_map,
    process_MS,
    process_SAR,
    read_img,
    read_tif,
)

logger = logging.getLogger(__name__)


def to_date(date_string: str) -> pd.Timestamp:
    """Convert date string to pandas timestamp."""
    return pd.Timestamp(date_string)


class SEN12MSCRTS:
    """Class to read the SEN12MS-CR-TS patches.

    Args:
        root_dir: Path to downloaded SEN12MS-CR-TS dataset
        split: Which splits to get data for: train, validation, test or all split.
            So far, the pixel-wise split only supports `all`.
        region: Which region(s) to obtain data for.
        cloud_masks: Type of cloud mask detector to run on optical data.
        sample_type: Whether to sample generic or cloudfree.
        n_input_samples: Number of input samples in time series if only part of the time series
            is used as an input.
        min_cov: Minimum cloud coverage, within range [0.0, 1.0].
            Only used for `sample_type`="cloudy_cloudfree".
        max_cov: Maximum cloud coverage, within range [0.0, 1.0].
            Only used for `sample_type`="cloudy_cloudfree".
        import_data_path: Path to importing the suppl. file specifying what time points to load
            for input and output
        export_data_path: Path to export the suppl. file specifying what time points to load
            for input and output

    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal["all"] = "all",  # TODO: so far, the pixel-wise split only supports `all`
        region: Literal["all", "africa", "america", "asiaEast", "asiaWest", "europa"] = "europa",
        cloud_masks: Literal[
            "cloud_cloudshadow_mask", "s2cloudless_map", "s2cloudless_mask"
        ] = "s2cloudless_mask",
        sample_type: Literal["generic", "cloudy_cloudfree"] = "cloudy_cloudfree",
        n_input_samples: int = 3,
        min_cov: float = 0.0,
        max_cov: float = 1.0,
        import_data_path: Path | None = None,
        export_data_path: Path | None = None,
    ) -> None:

        assert 0.0 <= min_cov <= 1.0, "min_cov needs to be in range [0.0, 1.0]."
        assert 0.0 <= max_cov <= 1.0, "max_cov needs to be in range [0.0, 1.0]."

        self.root_dir = root_dir  # set root directory which contains all ROI
        self.region = region  # region according to which the ROIs are selected
        self.ROI: dict[str, list[str]] = {
            "ROIs1158": ["106"],
            "ROIs1868": [
                "17",
                "36",
                "56",
                "73",
                "85",
                "100",
                "114",
                "119",
                "121",
                "126",
                "127",
                "139",
                "142",
                "143",
            ],
            "ROIs1970": [
                "20",
                "21",
                "35",
                "40",
                "57",
                "65",
                "71",
                "82",
                "83",
                "91",
                "112",
                "116",
                "119",
                "128",
                "132",
                "133",
                "135",
                "139",
                "142",
                "144",
                "149",
            ],
            "ROIs2017": [
                "8",
                "22",
                "25",
                "32",
                "49",
                "61",
                "63",
                "69",
                "75",
                "103",
                "108",
                "115",
                "116",
                "117",
                "130",
                "140",
                "146",
            ],
        }
        self.splits: dict[str, list[Path]] = {}
        if self.region == "all":
            all_ROI = [Path(key).joinpath(val) for key, vals in self.ROI.items() for val in vals]
            # official test split, across continents
            self.splits["test"] = [
                Path("ROIs1868").joinpath("119"),
                Path("ROIs1970").joinpath("139"),
                Path("ROIs2017").joinpath("108"),
                Path("ROIs2017").joinpath("63"),
                Path("ROIs1158").joinpath("106"),
                Path("ROIs1868").joinpath("73"),
                Path("ROIs2017").joinpath("32"),
                Path("ROIs1868").joinpath("100"),
                Path("ROIs1970").joinpath("132"),
                Path("ROIs2017").joinpath("103"),
                Path("ROIs1868").joinpath("142"),
                Path("ROIs1970").joinpath("20"),
                Path("ROIs2017").joinpath("140"),
            ]
            # official validation split, across continents
            self.splits["val"] = [
                Path("ROIs2017").joinpath("22"),
                Path("ROIs1970").joinpath("65"),
                Path("ROIs2017").joinpath("117"),
                Path("ROIs1868").joinpath("127"),
                Path("ROIs1868").joinpath("17"),
            ]
            # all remaining ROIs are used for training
            self.splits["train"] = [
                roi
                for roi in all_ROI
                if roi not in self.splits["val"] and roi not in self.splits["test"]
            ]
        elif self.region == "africa":
            self.splits["test"] = [
                Path("ROIs2017").joinpath("32"),
                Path("ROIs2017").joinpath("140"),
            ]
            self.splits["val"] = [Path("ROIs2017").joinpath("22")]
            self.splits["train"] = [
                Path("ROIs1970").joinpath("21"),
                Path("ROIs1970").joinpath("35"),
                Path("ROIs1970").joinpath("40"),
                Path("ROIs2017").joinpath("8"),
                Path("ROIs2017").joinpath("61"),
                Path("ROIs2017").joinpath("75"),
            ]
        elif self.region == "america":
            self.splits["test"] = [
                Path("ROIs1158").joinpath("106"),
                Path("ROIs1970").joinpath("132"),
            ]
            self.splits["val"] = [Path("ROIs1970").joinpath("65")]
            self.splits["train"] = [
                Path("ROIs1868").joinpath("36"),
                Path("ROIs1868").joinpath("85"),
                Path("ROIs1970").joinpath("82"),
                Path("ROIs1970").joinpath("142"),
                Path("ROIs2017").joinpath("49"),
                Path("ROIs2017").joinpath("116"),
            ]
        elif self.region == "asiaEast":
            self.splits["test"] = [
                Path("ROIs1868").joinpath("73"),
                Path("ROIs1868").joinpath("119"),
                Path("ROIs1970").joinpath("139"),
            ]
            self.splits["val"] = [Path("ROIs2017").joinpath("117")]
            self.splits["train"] = [
                Path("ROIs1868").joinpath("114"),
                Path("ROIs1868").joinpath("126"),
                Path("ROIs1868").joinpath("143"),
                Path("ROIs1970").joinpath("116"),
                Path("ROIs1970").joinpath("135"),
                Path("ROIs2017").joinpath("25"),
            ]
        elif self.region == "asiaWest":
            self.splits["test"] = [Path("ROIs1868").joinpath("100")]
            self.splits["val"] = [Path("ROIs1868").joinpath("127")]
            self.splits["train"] = [
                Path("ROIs1970").joinpath("57"),
                Path("ROIs1970").joinpath("83"),
                Path("ROIs1970").joinpath("112"),
                Path("ROIs2017").joinpath("69"),
                Path("ROIs1970").joinpath("115"),
                Path("ROIs1970").joinpath("130"),
            ]
        elif self.region == "europa":
            self.splits["test"] = [
                Path("ROIs2017").joinpath("63"),
                Path("ROIs2017").joinpath("103"),
                Path("ROIs2017").joinpath("108"),
                Path("ROIs1868").joinpath("142"),
                Path("ROIs1970").joinpath("20"),
            ]
            self.splits["val"] = [Path("ROIs1868").joinpath("17")]
            self.splits["train"] = [
                Path("ROIs1868").joinpath("56"),
                Path("ROIs1868").joinpath("121"),
                Path("ROIs1868").joinpath("139"),
                Path("ROIs1970").joinpath("71"),
                Path("ROIs1970").joinpath("91"),
                Path("ROIs1970").joinpath("119"),
                Path("ROIs1970").joinpath("128"),
                Path("ROIs1970").joinpath("133"),
                Path("ROIs1970").joinpath("144"),
                Path("ROIs1970").joinpath("149"),
                Path("ROIs2017").joinpath("146"),
            ]

        self.splits["all"] = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split

        self.modalities = ["S1", "S2"]
        self.time_points = range(30)  # 30 time steps
        # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.cloud_masks = cloud_masks
        # pick 'generic' or 'cloudy_cloudfree'
        self.sample_type = sample_type if self.cloud_masks is not None else "generic"
        # specifies the number of samples, if only part of the time series is used as an input
        self.n_input_t = n_input_samples

        if self.cloud_masks in ["s2cloudless_map", "s2cloudless_mask"]:
            self.cloud_detector: S2PixelCloudDetector | None = S2PixelCloudDetector(
                threshold=0.4, all_bands=True, average_over=4, dilation_size=2
            )
        else:
            self.cloud_detector = None

        self.import_data_path = import_data_path
        self.export_data_path = export_data_path
        if self.export_data_path:
            self.data_pairs = {}
        if self.import_data_path:
            # fetch time points as specified in the imported file,
            # expects arguments are set accordingly
            if self.import_data_path.is_dir():
                import_here = self.import_data_path.joinpath(
                    f"{self.n_input_t}_{self.split}_{self.cloud_masks}.npy"
                )
            else:
                import_here = self.import_data_path
            self.data_pairs = np.load(import_here, allow_pickle=True).item()
            logger.info(f"Importing data pairings for split {self.split} from {import_here}.")

        self.paths = self._get_paths()
        self.n_samples = len(self.paths)

        # raise a warning that no data has been found
        if not self.n_samples:
            self._throw_warn()

        self.min_cov, self.max_cov = min_cov, max_cov

    def _throw_warn(self) -> None:
        logger.warning(
            "No data samples found! Please use the following directory structure:\n"
            "\n"
            "path/to/your/SEN12MSCRTS/directory:\n"
            "├───ROIs1158\n"
            "├───ROIs1868\n"
            "├───ROIs1970\n"
            "│   ├───20\n"
            "│   ├───21\n"
            "│   │   ├───S1\n"
            "│   │   └───S2\n"
            "│   │       ├───0\n"
            "│   │       ├───1\n"
            "│   │       │   └─── ... *.tif files\n"
            "│   │       └───30\n"
            "│   ...\n"
            "└───ROIs2017\n"
            "\n"
            "Note: the data is provided by ROI geo-spatially separated and sensor modalities "
            "individually. You can simply merge the downloaded & extracted archives' "
            "subdirectories via 'mv */* .' in the parent directory to obtain the required "
            "structure specified above, which the dataloader expects."
        )

    # indexes all patches contained in the current data split
    def _get_paths(
        self,
    ) -> list[dict]:  # assuming for the same ROI+num, the patch numbers are the same
        """Collecting S1 & S2 filepaths for SEN12MS-CR-TS dataset."""
        logger.info(f"Processing paths for {self.split} split of region {self.region}.")
        paths = []
        for roi_dir, rois in self.ROI.items():
            for roi in tqdm(rois):
                roi_path = self.root_dir.joinpath(roi_dir, roi)
                # skip non-existent ROI or ROI not part of the current data split
                if (
                    not roi_path.is_dir()
                    or Path(roi_dir).joinpath(roi) not in self.splits[self.split]
                ):
                    continue
                path_s1_t, path_s2_t = (
                    [],
                    [],
                )
                for tdx in self.time_points:
                    # working with directory under time stamp tdx
                    path_s1_complete = roi_path.joinpath(self.modalities[0], str(tdx))
                    path_s2_complete = roi_path.joinpath(self.modalities[1], str(tdx))

                    # same as complete paths, truncating root directory's path
                    path_s1 = roi_path.joinpath(self.modalities[0], str(tdx))
                    path_s2 = roi_path.joinpath(self.modalities[1], str(tdx))

                    # get list of files which contains all the patches at time tdx
                    s1_t = natsorted(
                        [
                            path_s1.joinpath(f)
                            for f in path_s1_complete.iterdir()
                            if (path_s1_complete.joinpath(f).is_file() and ".tif" in str(f))
                        ]
                    )
                    s2_t = natsorted(
                        [
                            path_s2.joinpath(f)
                            for f in path_s2_complete.iterdir()
                            if (path_s2_complete.joinpath(f).is_file() and ".tif" in str(f))
                        ]
                    )

                    # same number of patches
                    assert len(s1_t) == len(s2_t)

                    # sort via file names according to patch number and store
                    path_s1_t.append(s1_t)
                    path_s2_t.append(s2_t)

                # for each patch of the ROI, collect its time points and make this one sample
                for pdx in range(len(path_s1_t[0])):
                    sample = {
                        "S1": [path_s1_t[tdx][pdx] for tdx in self.time_points],
                        "S2": [path_s2_t[tdx][pdx] for tdx in self.time_points],
                    }
                    paths.append(sample)

        return paths

    def _collect_locations(self, pdx: int) -> list[float]:
        """Collect locations of dataset patches."""
        s2_paths = [self.root_dir.joinpath(img_path) for img_path in self.paths[pdx]["S2"]]
        s2_tif = [read_tif(img_path) for img_path in s2_paths]
        coords = [list(tif.bounds) for tif in s2_tif]

        return coords[0]

    def _collect_patches(self, pdx: int) -> dict[str, dict | bool | list[str]]:
        s1_paths = [self.root_dir.joinpath(img_path) for img_path in self.paths[pdx]["S1"]]
        s2_paths = [self.root_dir.joinpath(img_path) for img_path in self.paths[pdx]["S2"]]

        s1_tif = [read_tif(img_path) for img_path in s1_paths]
        s2_tif = [read_tif(img_path) for img_path in s2_paths]
        coords = [list(tif.bounds) for tif in s2_tif]

        s1 = [process_SAR(read_img(img)) for img in s1_tif]
        s2 = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection

        masks = (
            None
            if not self.cloud_masks
            else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2]
        )
        # get statistics and additional meta information
        coverage = [np.mean(mask) for mask in masks]
        s1_dates = np.array(
            [
                to_date(img.name.split("/")[-1].split("_")[5]).strftime("%Y-%m-%d")
                for img in s1_paths
            ]
        )
        s2_dates = np.array(
            [
                to_date(img.name.split("/")[-1].split("_")[5]).strftime("%Y-%m-%d")
                for img in s2_paths
            ]
        )

        # generate data of ((cloudy_t1, cloudy_t2, ..., cloudy_tn), cloud-free) pairings
        # note: filtering the data (e.g. according to cloud coverage etc) may only use a fraction
        # of the data set
        # if you wish to train or test on additional samples,
        # then this filtering needs to be adjusted

        if self.sample_type == "cloudy_cloudfree":
            if self.import_data_path:
                # read indices
                inputs_idx = self.data_pairs[pdx]["input"]
                cloudless_idx = self.data_pairs[pdx]["target"]
                target_s1, target_s2, target_mask = (
                    np.array(s1)[cloudless_idx],
                    np.array(s2)[cloudless_idx],
                    np.array(masks)[cloudless_idx],
                )
                input_s1, input_s2, input_masks = (
                    np.array(s1)[inputs_idx],
                    np.array(s2)[inputs_idx],
                    np.array(masks)[inputs_idx],
                )
                coverage_match = True

            else:  # sample custom time points from the current patch space in the current split
                # sort observation indices according to cloud coverage, ascendingly
                coverage_idx = np.argsort(coverage)
                cloudless_idx = coverage_idx[0]
                # take the (earliest, in case of draw) least cloudy time point as target
                target_s1, target_s2, target_mask = (
                    np.array(s1)[cloudless_idx],
                    np.array(s2)[cloudless_idx],
                    np.array(masks)[cloudless_idx],
                )
                # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
                inputs_idx = [
                    pdx
                    for pdx, perc in enumerate(coverage)
                    if perc >= self.min_cov and perc <= self.max_cov
                ][: self.n_input_t]
                coverage_match = True  # assume the requested amount of cloud coverage is met

                if len(inputs_idx) < self.n_input_t:
                    # ... if not exists then take the first n_input_t samples (except target patch)
                    inputs_idx = [pdx for pdx in range(len(coverage)) if pdx != cloudless_idx][
                        : self.n_input_t
                    ]
                    # flag input samples that didn't meet the required cloud coverage
                    coverage_match = False
                input_s1, input_s2, input_masks = (
                    np.array(s1)[inputs_idx],
                    np.array(s2)[inputs_idx],
                    np.array(masks)[inputs_idx],
                )

            sample: dict = {
                "input": {
                    "S1": [input_s1],
                    "S2": [process_MS(img) for img in input_s2],
                    "masks": [input_masks],
                    "coverage": [np.mean(mask) for mask in input_masks],
                    "S1_dates": [s1_dates[idx] for idx in inputs_idx],
                    "S2_dates": [s2_dates[idx] for idx in inputs_idx],
                    "S1_path": [s1_paths[idx] for idx in inputs_idx],
                    "S2_path": [s2_paths[idx] for idx in inputs_idx],
                    "coords": [coords[idx] for idx in inputs_idx],
                },
                "target": {
                    "S1": [target_s1],
                    "S2": [process_MS(target_s2)],
                    "masks": [target_mask],
                    "coverage": [np.mean(target_mask)],
                    "S1_dates": [s1_dates[cloudless_idx]],
                    "S2_dates": [[cloudless_idx]],
                    "S1_path": [s1_paths[cloudless_idx]],
                    "S2_path": [s2_paths[cloudless_idx]],
                    "coords": [coords[cloudless_idx]],
                },
                "coverage bin": coverage_match,
            }

        elif (
            self.sample_type == "generic"
        ):  # this returns the whole, unfiltered sequence of S1 & S2 observations
            sample = {
                "S1": s1,
                "S2": [process_MS(img) for img in s2],
                "masks": masks,
                "coverage": coverage,
                "S1_dates": s1_dates,
                "S2_dates": s2_dates,
                "S1_path": [s1_paths[t_idx] for t_idx in self.time_points],
                "S2_path": [s2_paths[t_idx] for t_idx in self.time_points],
                "coords": coords,
            }
        return sample
