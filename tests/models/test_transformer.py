import pytest
import torch
from eurocropsml.dataset.base import DataItem

from eurocropsssl.models.transformer import TransformerConfig, TransformerModelBuilder


@pytest.fixture
def config() -> TransformerConfig:
    return TransformerConfig(
        n_heads=2,
        in_channels=18,
        d_model=64,
        dim_fc=128,
        num_layers=1,
        pos_enc_len=366,
    )


@pytest.fixture
def config_cross_attn() -> TransformerConfig:
    return TransformerConfig(
        n_heads=2,
        in_channels=18,
        d_model=64,
        dim_fc=128,
        num_layers=1,
        pos_enc_len=366,
        channel_encoding="cross_attn",
        channels=[
            "VV",
            "VH",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "8A",
            "09",
            "10",
            "11",
            "12",
            "2m_temperature",
            "total_precipitation",
            "NDVI",
        ],
    )


@pytest.fixture
def config_presto() -> TransformerConfig:
    return TransformerConfig(
        n_heads=4,
        in_channels=18,
        d_model=128,
        dim_fc=2048,
        num_layers=1,
        pos_enc_len=366,
        channel_encoding="presto",
        channels=[
            "VV",
            "VH",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "8A",
            "09",
            "10",
            "11",
            "12",
            "2m_temperature",
            "total_precipitation",
            "NDVI",
        ],
    )


@pytest.fixture
def model_builder(config: TransformerConfig) -> TransformerModelBuilder:
    return TransformerModelBuilder(config=config)


@pytest.fixture
def model_builder_cross_attn(
    config_cross_attn: TransformerConfig,
) -> TransformerModelBuilder:
    return TransformerModelBuilder(config=config_cross_attn)


@pytest.fixture
def model_builder_presto(config_presto: TransformerConfig) -> TransformerModelBuilder:
    return TransformerModelBuilder(config=config_presto)


def test_classification_forward(
    model_builder: TransformerModelBuilder,
    timeseries_data: DataItem,
    device: torch.device,
) -> None:
    num_classes = 10
    model = model_builder.build_classification_model(num_classes, device)

    out = model(timeseries_data)
    assert out.size() == (timeseries_data.data.size(0), num_classes)


def test_self_supervised_forward(
    model_builder: TransformerModelBuilder,
    timeseries_data: DataItem,
    device: torch.device,
) -> None:
    model = model_builder.build_self_supervised_model(device)

    out = model(timeseries_data)

    assert out.size() == timeseries_data.data.size()


def test_self_supervised_forward_cross_attn(
    model_builder_cross_attn: TransformerModelBuilder,
    timeseries_data: DataItem,
    device: torch.device,
) -> None:
    model = model_builder_cross_attn.build_self_supervised_model(device)

    out = model(timeseries_data)

    assert out.size() == timeseries_data.data.size()


def test_self_supervised_forward_presto(
    model_builder_presto: TransformerModelBuilder,
    timeseries_data: DataItem,
    device: torch.device,
) -> None:
    model = model_builder_presto.build_self_supervised_model(device)

    out = model(timeseries_data)

    assert out.size() == timeseries_data.data.size()


def test_reset_head(
    model_builder: TransformerModelBuilder,
    timeseries_data: DataItem,
    device: torch.device,
) -> None:
    num_classes = 10
    model = model_builder.build_classification_model(num_classes, device)

    out = model(timeseries_data)
    assert out.size() == (timeseries_data.data.size(0), num_classes)

    num_new_classes = 7
    model_builder.reset_head(model, num_new_classes)
    out = model(timeseries_data)

    assert out.size() == (timeseries_data.data.size(0), num_new_classes)
