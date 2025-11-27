from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from eurocropsssl.dataset.era5.download import download_with_retry


@pytest.fixture
def fake_path(tmp_path: Path) -> Path:
    return tmp_path / "era5_2018_01_01.grb"


def test_successful_download(monkeypatch: pytest.MonkeyPatch, fake_path: Path) -> None:
    mock_client = mock.Mock()
    monkeypatch.setattr("eurocropsssl.dataset.era5.download.cdsapi.Client", lambda: mock_client)

    # ensure file does not exist yet
    assert not fake_path.exists()

    download_with_retry(
        output_path=fake_path.parent,
        bbox=(50.0, 5.0, 45.0, 10.0),
        climate_variables=["2m_temperature", "total_precipitation"],
        dates=(2018, 1, 1),
    )

    # retrieve method (API call) was called once
    mock_client.retrieve.assert_called_once()


def test_skip_if_file_exists(monkeypatch: pytest.MonkeyPatch, fake_path: Path) -> None:
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("already downloaded")  # Pretend file already exists

    mock_client = mock.Mock()
    monkeypatch.setattr("eurocropsssl.dataset.era5.download.cdsapi.Client", lambda: mock_client)

    download_with_retry(
        output_path=fake_path.parent,
        bbox=(50.0, 5.0, 45.0, 10.0),
        climate_variables=["2m_temperature", "total_precipitation"],
        dates=(2018, 1, 1),
    )
    # no API call since file exists already
    mock_client.retrieve.assert_not_called()


def test_retry_on_failure(monkeypatch: pytest.MonkeyPatch, fake_path: Path) -> None:
    mock_client = mock.Mock()
    call_count = {"count": 0}

    def flaky_retrieve(*args: Any, **kwargs: Any) -> None:
        # fail on first two attempts (count<2)
        if call_count["count"] < 2:
            call_count["count"] += 1
            raise Exception("Temporary failure")
        return None

    mock_client.retrieve.side_effect = flaky_retrieve
    monkeypatch.setattr("eurocropsssl.dataset.era5.download.cdsapi.Client", lambda: mock_client)

    download_with_retry(
        output_path=fake_path.parent,
        bbox=(50.0, 5.0, 45.0, 10.0),
        climate_variables=["2m_temperature", "total_precipitation"],
        dates=(2018, 1, 1),
        max_attempts=3,
        initial_delay=0.01,  # Speed up test
    )
    # succeed on third attemp
    assert mock_client.retrieve.call_count == 3


def test_fails_after_retries(monkeypatch: pytest.MonkeyPatch, fake_path: Path) -> None:
    mock_client = mock.Mock()
    mock_client.retrieve.side_effect = Exception("Persistent failure")
    monkeypatch.setattr("eurocropsssl.dataset.era5.download.cdsapi.Client", lambda: mock_client)

    with pytest.raises(Exception, match="Persistent failure"):
        download_with_retry(
            output_path=fake_path.parent,
            bbox=(50.0, 5.0, 45.0, 10.0),
            climate_variables=["2m_temperature", "total_precipitation"],
            dates=(2018, 1, 1),
            max_attempts=2,
            initial_delay=0.01,
        )
    # ensure that max_attempts=2 worked
    assert mock_client.retrieve.call_count == 2
