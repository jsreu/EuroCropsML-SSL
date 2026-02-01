from __future__ import annotations

import pytest
import torch
from eurocropsml.dataset.base import DataItem


@pytest.fixture(
    params=[
        {"dates": torch.cat([torch.arange(0, 50, dtype=torch.long).unsqueeze(0)] * 4)},
        {
            "dates": torch.cat([torch.arange(0, 50, dtype=torch.long).unsqueeze(0)] * 4),
            "mask": torch.randint(0, 2, (4, 50, 18), dtype=torch.bool),  # 3D random mask
        },
    ],
    ids=["without mask", "with mask"],
)
def timeseries_meta_data(request) -> dict[str, torch.Tensor]:  # type: ignore
    return request.param  # type: ignore


@pytest.fixture
def timeseries_data(timeseries_meta_data: dict[str, torch.Tensor] | None) -> DataItem:
    x = torch.randn((4, 50, 18))  # batch size 4

    return DataItem(data=x, meta_data=timeseries_meta_data)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
