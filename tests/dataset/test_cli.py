from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel
from typer.testing import CliRunner, Typer


class MockConfig(BaseModel):
    name: str
    value: int


# ERA5DatasetConfig
class MockERA5Config(BaseModel):
    name: str
    value: int


# SEN12MSCRTSDatasetConfig
class MockSEN12Config(BaseModel):
    name: str
    value: int
    raw_data_dir: str = "/mock/data/dir"
    use_sentinel1_data: bool = True
    region: str = "test_region"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config(monkeypatch: Any, tmp_path: Path) -> MockConfig:
    config = MockConfig(name="test_dataset", value=1)

    config_dir = tmp_path / "experiments" / "common" / "pretrain_dataset"
    config_path = config_dir / f"{config.name}.yaml"

    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config.model_dump()))

    monkeypatch.setenv("EUROCROPS_SSL_EXPERIMENT_DIR", str(tmp_path / "experiments"))

    return config


@pytest.fixture
def app(config: MockConfig) -> Typer:
    from eurocropsssl.dataset.cli import build_dataset_app

    return cast(
        Typer,
        build_dataset_app(
            dataset_name=config.name,
            config_class=MockConfig,
        ),
    )


@pytest.fixture
def era5_app(monkeypatch: Any, tmp_path: Path) -> Typer:
    from eurocropsssl.dataset.cli import build_dataset_app

    # create config file
    config = MockERA5Config(name="era5_dataset", value=1)

    config_dir = tmp_path / "experiments" / "common" / "pretrain_dataset"
    config_path = config_dir / f"{config.name}.yaml"

    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config.model_dump()))

    monkeypatch.setenv("EUROCROPS_SSL_EXPERIMENT_DIR", str(tmp_path / "experiments"))

    # mock for isinstance check
    original_isinstance = isinstance

    def mock_isinstance(obj: BaseModel, class_or_tuple: Any) -> bool:
        if obj.__class__.__name__ == "MockERA5Config":
            if hasattr(class_or_tuple, "__name__"):
                if class_or_tuple.__name__ == "ERA5DatasetConfig":
                    return True
            elif isinstance(class_or_tuple, tuple):
                for cls in class_or_tuple:
                    if hasattr(cls, "__name__") and cls.__name__ == "ERA5DatasetConfig":
                        return True
        return original_isinstance(obj, class_or_tuple)

    monkeypatch.setattr("builtins.isinstance", mock_isinstance)

    # build and return dataset app
    return cast(
        Typer,
        build_dataset_app(
            dataset_name=config.name,
            config_class=MockERA5Config,
        ),
    )


@pytest.fixture
def sen12_app(monkeypatch: Any, tmp_path: Path) -> Typer:
    from eurocropsssl.dataset.cli import build_dataset_app

    # create config file
    config = MockSEN12Config(name="sen12_dataset", value=1)

    config_dir = tmp_path / "experiments" / "common" / "pretrain_dataset"
    config_path = config_dir / f"{config.name}.yaml"

    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config.model_dump()))

    monkeypatch.setenv("EUROCROPS_SSL_EXPERIMENT_DIR", str(tmp_path / "experiments"))

    # mock for isinstance check
    original_isinstance = isinstance

    def mock_isinstance(obj: BaseModel, class_or_tuple: Any) -> bool:
        if obj.__class__.__name__ == "MockSEN12Config":
            if hasattr(class_or_tuple, "__name__"):
                if class_or_tuple.__name__ == "SEN12MSCRTSDatasetConfig":
                    return True
                elif class_or_tuple.__name__ == "ERA5DatasetConfig":
                    return False
            elif isinstance(class_or_tuple, tuple):
                for cls in class_or_tuple:
                    if hasattr(cls, "__name__") and cls.__name__ == "SEN12MSCRTSDatasetConfig":
                        return True
        return original_isinstance(obj, class_or_tuple)

    monkeypatch.setattr("builtins.isinstance", mock_isinstance)

    # build and return dataset app
    return cast(
        Typer,
        build_dataset_app(
            dataset_name=config.name,
            config_class=MockSEN12Config,
        ),
    )


def test_print_config(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config"], catch_exceptions=False)

    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()


def test_print_config_overrides(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config", "value=2"], catch_exceptions=False)
    config.value = 2
    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()


def test_download_data_era5(era5_app: Typer, runner: CliRunner, monkeypatch: Any) -> None:
    # mock Path.exists to return True for .cdsapirc check
    original_exists = Path.exists

    def mock_exists(self: Path) -> bool:
        if str(self).endswith(".cdsapirc"):
            return True
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", mock_exists)

    # mock download_era5 function
    mock_download_era5 = MagicMock()
    monkeypatch.setattr("eurocropsssl.dataset.cli.download_era5", mock_download_era5)

    # call the command
    result = runner.invoke(era5_app, ["download"], catch_exceptions=False)

    assert result.exit_code == 0
    mock_download_era5.assert_called_once()


def test_download_data_era5_missing_config(
    era5_app: Typer, runner: CliRunner, monkeypatch: Any
) -> None:
    # Mock Path.exists to return False for .cdsapirc
    original_exists = Path.exists

    def mock_exists(self: Path) -> bool:
        if str(self).endswith(".cdsapirc"):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", mock_exists)

    # Mock download_era5 function
    mock_download_era5 = MagicMock()
    monkeypatch.setattr("eurocropsssl.dataset.cli.download_era5", mock_download_era5)

    # Call the command - this should fail
    result = runner.invoke(era5_app, ["download"], catch_exceptions=True)

    assert result.exit_code != 0

    # check if file not found
    assert isinstance(result.exception, FileNotFoundError)
    assert "CDSAPI configuration file not found at " in str(result.exception)
    mock_download_era5.assert_not_called()


def test_download_data_sen12mscrts(
    sen12_app: Typer, runner: CliRunner, monkeypatch: Any, tmp_path: Path
) -> None:
    # create temporary directory and bash script
    script_dir = tmp_path / "sen12mscrts"
    script_dir.mkdir()
    script_path = script_dir / "dl_data.sh"

    # create mock bash script
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Mock download script\n")
        f.write("echo 'Arguments received: $@'\n")

    # make it executable
    script_path.chmod(0o755)

    # mock the __file__ resolution to point to tmp directory
    monkeypatch.setattr("eurocropsssl.dataset.cli.__file__", str(tmp_path / "cli.py"))

    # mock subprocess.run
    mock_subprocess_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # call the command
    result = runner.invoke(sen12_app, ["download"], catch_exceptions=False)

    # assertions
    assert result.exit_code == 0
    mock_subprocess_run.assert_called_once()

    # check the arguments
    args = mock_subprocess_run.call_args[0][0]
    assert args[0] == "bash"

    # convert args to string for easier checking
    args_str = str(args)
    assert "test_region" in args_str
    assert "true" in args_str.lower() or "false" in args_str.lower()
    assert str(script_path) in args_str or script_path.name in args_str


def test_cli_preprocess(app: Typer, runner: CliRunner, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "eurocropsssl.dataset.cli.generate_pixel_timeseries",
        lambda config: print("Mock preprocess"),
    )

    result = runner.invoke(app, ["preprocess"], catch_exceptions=False)
    assert result.exit_code == 0
