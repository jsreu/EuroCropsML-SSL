import logging
import multiprocessing as mp_orig
from functools import partial
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt
from tqdm import tqdm

from eurocropsssl.dataset.era5.config import ERA5DatasetConfig
from eurocropsssl.dataset.era5.preprocess import extract_values, load_era5_data

logger = logging.getLogger(__name__)


def _process_column(
    era5_dir: Path,
    merged_df_wkt: pd.DataFrame,
    output_dir: Path,
    chunk_size: int,
    normalize: bool,
    date: str,
) -> str:
    """Process a single ERA5 date column by extracting values for each parcel."""

    output_file = output_dir / f"{date}.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        return date

    merged_df = merged_df_wkt.copy()
    merged_df["geometry"] = merged_df["geometry_wkt"].apply(wkt.loads)
    merged_df = gpd.GeoDataFrame(merged_df, geometry="geometry", crs="EPSG:4326")

    ds_an, ds_fc = load_era5_data(date, era5_dir)

    projected_geom = merged_df.geometry.to_crs(epsg=3035)
    centroids_projected = projected_geom.centroid
    # convert centroids back to WGS84 (EPSG:4326) for coordinates
    centroids = centroids_projected.to_crs(epsg=4326)
    lon_lat_pairs = [(point.x, point.y) for point in centroids]
    parcel_ids = merged_df["parcel_id"].values

    parquet_writer = None

    for i in range(0, len(lon_lat_pairs), chunk_size):
        chunk_pairs = lon_lat_pairs[i : i + chunk_size]
        chunk_ids = parcel_ids[i : i + chunk_size]
        era5_values = []

        for lon, lat in chunk_pairs:
            bbox = (lon, lat, lon, lat)
            values = extract_values(ds_an, ds_fc, bbox, normalize)
            era5_values.append(values)

        chunk_df = pd.DataFrame(era5_values, columns=["2m_temperature", "total_precipitation"])
        chunk_df["parcel_id"] = chunk_ids
        table = pa.Table.from_pandas(chunk_df)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_file, table.schema)

        parquet_writer.write_table(table)

    ds_an.close()
    ds_fc.close()

    if parquet_writer:
        parquet_writer.close()

    return date


def extract_era5_data(
    config: ERA5DatasetConfig,
    input_geojson: Path,
    parcel_meta_path: list[Path],
    output_dir: Path,
    chunk_size: int = 10000,
    normalize: bool = False,
) -> None:
    """Extracts ERA5 values.

    Args:
        config: ERA5DatasetConfig
        input_geojson: Path to the input GeoJSON file.
        parcel_meta_path: List of paths to Parquet files (for S1 and/or S2) with parcel metadata.
        output_dir: Directory where output files will be stored.
        chunk_size: Chunk size used for multiprocessing.
        normalize: Whether to normalize the values.
    """

    era5_dir = config.raw_data_dir
    max_workers = config.workers

    gdf = gpd.read_file(input_geojson)
    # collect all parcel from S1/S2
    parquet_dfs = [pd.read_parquet(file_path) for file_path in parcel_meta_path]
    parquet_df = pd.concat(parquet_dfs, ignore_index=False, join="outer")
    # drop duplicates based on 'parcel_id' column, keeping the first occurrence
    parquet_df = parquet_df.drop_duplicates(subset=["parcel_id"])
    merged_df = gdf.merge(parquet_df, on="parcel_id")

    if not merged_df.crs.is_geographic:
        merged_df = merged_df.to_crs(epsg=4326)

    merged_df["geometry_wkt"] = merged_df["geometry"].apply(lambda geom: geom.wkt)
    merged_df_wkt = merged_df.copy()

    year: int = cast(int, config.year)
    days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    dates = pd.date_range(f"{year}-01-01", periods=days, freq="D").strftime("%Y-%m-%d")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    func = partial(_process_column, era5_dir, merged_df_wkt, output_dir, chunk_size, normalize)

    with mp_orig.Pool(processes=max_workers) as pool:
        te = tqdm(total=len(dates), desc="Processing ERA5 dates", position=0)
        for _ in pool.imap_unordered(func, dates):
            te.update()
        te.close()

    logger.info(f"Extraction complete. ERA5 values saved in: {output_dir}")


def save_era5_npz(
    config: ERA5DatasetConfig,
    parcel_meta_path: list[Path],
    era5_dir: Path,
    output_dir: Path,
) -> None:
    """Converts per-day ERA5 Parquet files into compressed NumPy (.npz) format for each parcel.

    Args:
        config: ERA5DatasetConfig
        parcel_meta_path: List of paths to Parquet files (for S1 and/or S2) with parcel metadata.
        era5_dir: Directory containing daily ERA5 Parquet files.
        output_dir: Directory to save the output .npz files.
    """

    parquet_dfs = [pd.read_parquet(file_path) for file_path in parcel_meta_path]
    parquet_df = pd.concat(parquet_dfs, ignore_index=False, join="outer")
    # drop duplicates based on 'parcel_id' column, keeping the first occurrence
    parquet_df = parquet_df.drop_duplicates(subset=["parcel_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_files = sorted(
        [f for f in era5_dir.glob("*.parquet") if f.name.startswith(f"{config.year}-")]
    )
    date_list = [f.stem for f in era5_files]

    logger.info("Loading daily ERA5 parcel files...")
    era5_data_by_date = {}
    for file in tqdm(era5_files, desc="Reading ERA5 files"):
        date = file.stem
        era5_data_by_date[date] = pd.read_parquet(file).set_index("parcel_id")

    logger.info("Processing parcels...")
    for _, row in tqdm(parquet_df.iterrows(), total=len(parquet_df), desc="Saving .npz files"):
        parcel_id = row["parcel_id"]
        nuts3 = row["nuts3"]
        crop_class = row["EC_hcat_c"]

        temps = []
        precs = []

        for date in date_list:
            daily_df = era5_data_by_date.get(date)
            if daily_df is not None and parcel_id in daily_df.index:
                record = daily_df.loc[parcel_id]
                temps.append(record["2m_temperature"])
                precs.append(record["total_precipitation"])
            else:
                temps.append(np.nan)
                precs.append(np.nan)

        np.savez_compressed(
            output_dir / f"{nuts3}_{parcel_id}_{crop_class}.npz",
            temperature=np.array(temps, dtype=np.float32),
            precipitation=np.array(precs, dtype=np.float32),
            dates=np.array(date_list),
        )

    logger.info(f"Processing complete. ERA5 .npz files saved in: {output_dir}")
