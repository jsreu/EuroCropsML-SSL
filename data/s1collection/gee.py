import datetime
import glob
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Optional

import ee

logger = logging.getLogger("gee_processing")


class RateLimiter:
    """Add a rate limiter class to manage concurrent requests."""

    def __init__(self, max_requests: int = 20, time_window: int = 60):
        self.queue: Queue = Queue()
        self.lock = Lock()
        self.max_requests = max_requests
        self.time_window = time_window

    def _acquire(self) -> None:
        current_time = time.time()
        with self.lock:
            while not self.queue.empty() and current_time - self.queue.queue[0] > self.time_window:
                self.queue.get()

            if self.queue.qsize() >= self.max_requests:
                oldest_request = self.queue.queue[0]
                sleep_time = oldest_request + self.time_window - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.queue.put(current_time)


def find_last_processed_raster(log_dir: Path) -> Optional[int]:
    """Read through all log files to find the highest raster index that was successfully processed.

    Returns the highest raster index found, or None if no rasters were completed.
    """
    log_files = sorted(glob.glob(str(log_dir / "gee_processing_*.log")), reverse=True)

    if not log_files:
        return None

    pattern = r"All files for raster (\d+) are being exported\."
    highest_raster_idx = -1

    # Check each log file
    for log_file in log_files:
        with open(log_file, "r") as f:
            content = f.read()

        # Find all matches in this file
        matches = re.finditer(pattern, content)
        for match in matches:
            raster_idx = int(match.group(1))
            highest_raster_idx = max(highest_raster_idx, raster_idx)

    return highest_raster_idx if highest_raster_idx >= 0 else None


def get_s1_collection(start_date: str, end_date: str, country_geometry: str) -> ee.ImageCollection:
    """Get collection of S1 files."""
    logger.info(f"Collecting Sentinel-1 tiles from {start_date} to {end_date}...")
    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(country_geometry)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .sort("system:time_start")
    )

    collection_size = s1_collection.size().getInfo()
    logger.info(f"Found {collection_size} images.")
    return s1_collection


def _process_chunk(chunk_data: tuple) -> bool:
    image, parcels, raster_idx, chunk_start, chunk_size, rate_limiter = chunk_data
    try:
        rate_limiter._acquire()

        date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        s1_db = image.select(["VV", "VH"])
        img_projection = image.select("VV").projection()

        parcels_reprojected = parcels.map(
            lambda feature: feature.setGeometry(feature.geometry().transform(img_projection, 0.01))
        )

        parcel_list = parcels_reprojected.toList(chunk_size, chunk_start)
        chunk = ee.FeatureCollection(parcel_list)

        chunk_stats = s1_db.reduceRegions(
            collection=chunk, reducer=ee.Reducer.median(), scale=10
        ).filter(ee.Filter.notNull(["VV", "VH"]))

        chunk_stats = chunk_stats.map(lambda feature: feature.set({"date": date_str}))

        export_task = ee.batch.Export.table.toCloudStorage(
            collection=chunk_stats,
            description=f"raster_{raster_idx}_chunk_{chunk_start}",
            fileFormat="CSV",
            bucket="eurocropsml",
            fileNamePrefix=f"Estonia_S1/raster_{raster_idx}_chunk_{chunk_start}",
        )
        export_task.start()

        logger.info(f"Successfully processed chunk {chunk_start} for raster {raster_idx}")
        return True

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_start} for raster {raster_idx}: {str(e)}")
        return False


def _process_image(
    image: ee.Image,
    parcels: ee.FeatureCollection,
    raster_idx: int,
    max_workers: int = 4,
    chunk_size: int = 2500,
) -> None:
    logger.info(f"Processing raster {raster_idx}...")

    tile_geometry = image.select("VV").geometry()
    true_intersecting_parcels = parcels.filterBounds(tile_geometry)
    total_intersect = true_intersecting_parcels.size().getInfo()

    logger.info(f"Raster {raster_idx} has {total_intersect} intersecting polygons.")

    rate_limiter = RateLimiter(
        max_requests=30, time_window=60
    )  # Adjust these values based on GEE limits

    chunks = []
    for start in range(0, total_intersect, chunk_size):
        chunks.append(
            (image, true_intersecting_parcels, raster_idx, start, chunk_size, rate_limiter)
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk_data) for chunk_data in chunks]

        # Wait for all chunks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Chunk processing failed: {str(e)}")

    logger.info(f"All files for raster {raster_idx} are being exported.")


def process_with_recovery(
    start_date: str, end_date: str, current_dir: Path, max_workers: int = 4
) -> None:
    """Main processing function with automatic recovery and parallelization."""
    ee.Initialize()

    log_dir = current_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    parcels = ee.FeatureCollection("projects/ee-eurocropsml/assets/EE_2021_EC21")
    parcels = parcels.select(["pollu_id"])

    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    estonia = countries.filter(ee.Filter.eq("ADM0_NAME", "Estonia"))

    if not current_dir.joinpath("s1_orbit_info.json").exists():
        raise FileNotFoundError("s1_orbit_info.json file is missing.")

    while True:
        try:
            last_raster = find_last_processed_raster(log_dir)
            start_idx = (last_raster + 1) if last_raster is not None else 0

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"gee_processing_{timestamp}.log"

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )

            s1_collection = get_s1_collection(start_date, end_date, estonia.geometry())
            s1_list = s1_collection.toList(s1_collection.size())
            total_images = s1_list.size().getInfo()
            orbits = s1_collection.aggregate_array("orbitProperties_pass").getInfo()

            # Process the orbit properties
            orbit_info = {}
            for raster_idx in range(total_images):
                orbit_info[raster_idx] = {"orbit": orbits[raster_idx]}

            with open(current_dir.joinpath("s1_orbit_info.json"), "w") as f:
                json.dump(orbit_info, f, indent=2)
            total_images = s1_list.size().getInfo()

            if last_raster is not None:
                logger.info(f"Recovering from previous run. Starting from raster {start_idx}")
            else:
                logger.info("Starting new processing run")

            for i in range(start_idx, total_images):
                image = ee.Image(s1_list.get(i))
                _process_image(image, parcels, i, max_workers=max_workers)
                time.sleep(2)  # Small delay between rasters

            logger.info("Processing completed successfully")
            break

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            logger.info("Waiting 60 seconds before retrying...")
            time.sleep(60)
            continue


if __name__ == "__main__":
    start_date = "2021-01-01"
    end_date = "2021-12-31"

    current_dir = Path(__file__).resolve().parent
    process_with_recovery(start_date, end_date, current_dir, max_workers=16)
