from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from eurocropsssl.models.base import Model, ModelBuilder
from tests.utils import DenseNNBuilder

NUM_CLASSES = 10


@pytest.fixture
def model_builder() -> ModelBuilder:
    return cast(ModelBuilder, DenseNNBuilder(in_size=10))


@pytest.fixture
def classification_model(model_builder: ModelBuilder) -> Model:
    return model_builder.build_classification_model(
        num_classes=NUM_CLASSES, device=torch.device("cpu")
    )


@pytest.fixture
def self_supervised_model(model_builder: ModelBuilder) -> Model:
    return model_builder.build_self_supervised_model(device=torch.device("cpu"))


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path.joinpath("model_checkpoint")
    checkpoint.mkdir()
    return checkpoint


@pytest.mark.parametrize("model", ["classification_model", "self_supervised_model"])
@pytest.mark.parametrize("load_head", [True, False])
def test_save_load_model(model: Model, checkpoint: Path, load_head: bool, request: Any) -> None:
    if model == "self_supervised_model":
        with pytest.raises(NotImplementedError):
            request.getfixturevalue(model)
    else:
        model = request.getfixturevalue(model)
        model.save(checkpoint)

        loaded_model = deepcopy(model)
        loaded_model.load(checkpoint=checkpoint, load_head=load_head)

        if load_head:
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert (p1 == p2).all()
        else:
            for p1, p2 in zip(model.backbone.parameters(), loaded_model.backbone.parameters()):
                assert (p1 == p2).all()
