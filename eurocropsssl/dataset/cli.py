import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Type, TypeVar

import typer
from eurocropsml.acquisition.config import EuroCropsCountryConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel

from eurocropsssl.dataset.era5.config import ERA5DatasetConfig
from eurocropsssl.dataset.era5.download import download_era5
from eurocropsssl.dataset.eurocrops.add_era5 import extract_era5_data, save_era5_npz
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.dataset.sen12mscrts.preprocess import generate_pixel_timeseries
from eurocropsssl.settings import Settings

logger = logging.getLogger(__name__)

datasets_app = typer.Typer(name="datasets")

ConfigT = TypeVar("ConfigT", bound=BaseModel)
OverridesT = Optional[list[str]]


def build_dataset_app(dataset_name: str, config_class: Type[ConfigT]) -> typer.Typer:
    """Build cli component for preparing datasets."""

    config_dir = Settings().experiment_dir.joinpath("common", "pretrain_dataset")

    app = typer.Typer(name=dataset_name)

    def build_config(overrides: OverridesT) -> ConfigT:
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            if overrides is None:
                overrides = []
            composed_config = compose(config_name=dataset_name, overrides=overrides)
        config = config_class(**OmegaConf.to_object(composed_config))  # type: ignore[arg-type]
        return config

    @app.command(name="config")
    def print_config(
        overrides: OverridesT = typer.Argument(None, help="Overrides to config"),
    ) -> None:
        """Print currently used config."""
        config = build_config(overrides)
        print(OmegaConf.to_yaml(json.loads(config.json())))

    @app.command(name="download")
    def download_data(
        overrides: OverridesT = typer.Argument(None, help="Overrides to preprocess config"),
    ) -> None:
        config = build_config(overrides)

        if isinstance(config, SEN12MSCRTSDatasetConfig):
            data_dir: str = str(config.raw_data_dir)
            use_sentinel1_data: str = str(config.use_sentinel1_data).lower()
            region: str = config.region

            bash_script: str = str(Path(__file__).resolve().parent / "sen12mscrts" / "dl_data.sh")
            subprocess.run(["bash", bash_script, region, use_sentinel1_data, data_dir])

        elif isinstance(config, ERA5DatasetConfig):
            cdsapi_config_path = Path.home() / ".cdsapirc"
            if not cdsapi_config_path.exists():
                raise FileNotFoundError(
                    f"CDSAPI configuration file not found at {cdsapi_config_path}.\n"
                    "Please configure the CDS API first by following the instructions at:\n"
                    "https://cds.climate.copernicus.eu/how-to-api"
                )
            download_era5(config)

    @app.command(name="preprocess")
    def preprocess_data(
        overrides: OverridesT = typer.Argument(None, help="Overrides to preprocess config"),
    ) -> None:
        config = build_config(overrides)
        if isinstance(config, SEN12MSCRTSDatasetConfig):
            generate_pixel_timeseries(config=config)

        elif isinstance(config, ERA5DatasetConfig):
            data_dir = Path(Settings().data_dir)
            country = config.base_dataset.split("_")[-1].capitalize()
            country_code: str = EuroCropsCountryConfig().countries[country]["country_code"]
            raw_data_dir: Path = data_dir / "raw_data"
            parquet_files: list[Path] = [
                filepath.joinpath(str(config.year), f"{country_code}.parquet")
                for filepath in raw_data_dir.iterdir()
                if filepath.name in ["S1", "S2"]
            ]
            geom_file = raw_data_dir / "geometries" / str(config.year) / f"{country_code}.geojson"
            era5_days = data_dir / "era5_days"
            era5_days.mkdir(parents=True, exist_ok=True)
            era5_npz = data_dir / "era5_npz"
            era5_npz.mkdir(parents=True, exist_ok=True)

            if not parquet_files:
                logger.error(
                    f"Meta-data file for {config.base_dataset} is missing. "
                    f"Please upload it under {str(raw_data_dir)}."
                )
            if not geom_file.exists():
                logger.error(
                    f"Geometry file for {config.base_dataset} is missing. "
                    f"Please upload it under {str(raw_data_dir / 'geometries' / str(config.year))}."
                )

            extract_era5_data(
                config,
                geom_file,
                parquet_files,
                era5_days,
                config.chunk_size,
                config.normalize,
            )
            save_era5_npz(config, parquet_files, era5_days, era5_npz)

    return app


datasets_app.add_typer(build_dataset_app("era5", ERA5DatasetConfig))
datasets_app.add_typer(build_dataset_app("sen12mscrts", SEN12MSCRTSDatasetConfig))
