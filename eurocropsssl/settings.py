from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

EPSILON = 1e-6

ROOT_DIR = Path(__file__).parents[1]


class Settings(BaseSettings):
    """Global settings.

    Args:
        data_dir: Main data directory.
        era5_data_dir: Directory in which ERA5 data is stored.
        sen12ms_data_dir: Directory in which SEN12MSCRTS data is stored.
        experiment_dir: Directory in which experiment runs are stored.
        mlflow_uri: URI to mlflow tracking server.
        disable_cudnn: Whether to disable cuDNN.
            This may be necessary when training RNNs.
    """

    data_dir: Path = Field(Path("data"), validation_alias="EUROCROPS_SSL_DATA_DIR")

    era5_data_dir: Path = Field(Path("era5"), validation_alias="EUROCROPS_SSL_ERA5_DATA_DIR")

    sen12ms_data_dir: Path = Field(
        Path("sen12ms"), validation_alias="EUROCROPS_SSL_SEN12MS_DATA_DIR"
    )
    experiment_dir: Path = Field(
        Path("experiments"), validation_alias="EUROCROPS_SSL_EXPERIMENT_DIR"
    )
    mlflow_uri: str = Field(
        "file:./mlruns",  # Local file storage by default
        validation_alias="EUROCROPS_SSL_MLFLOW_URI",
    )
    disable_cudnn: bool = Field(False, validation_alias="EUROCROPS_SSL_DISABLE_CUDNN")
    seed: int = Field(42, validation_alias="EUROCROPS_SSL_SEED")

    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        """Validate all paths in the correct order."""
        # First handle data_dir
        if not self.data_dir.is_absolute():
            self.data_dir = ROOT_DIR.joinpath(self.data_dir)

        # Then handle other paths
        if not self.experiment_dir.is_absolute():
            self.experiment_dir = ROOT_DIR.joinpath(self.experiment_dir)

        # Handle paths that depend on data_dir
        for field in ["sen12ms_data_dir", "era5_data_dir"]:
            path = getattr(self, field)
            if not path.is_absolute():
                setattr(self, field, self.data_dir.joinpath(path))

        return self
