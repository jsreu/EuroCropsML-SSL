import json
import logging
from functools import partial, reduce
from operator import add
from pathlib import Path
from typing import Literal

import torch.nn as nn
from eurocropsml.dataset.base import TransformDataset
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig

from eurocropsssl.dataset.base import custom_collate_fn
from eurocropsssl.dataset.eurocrops.config import EuroCropsDatasetConfigSSL
from eurocropsssl.dataset.eurocrops.dataset import EuroCropsDatasetSSL
from eurocropsssl.dataset.eurocrops.utils import MMapStoreERA5
from eurocropsssl.dataset.task import Task
from eurocropsssl.settings import Settings
from eurocropsssl.train.utils import get_metrics

logger = logging.getLogger(__name__)


def load_dataset_split(
    mode: Literal["pretraining", "finetuning"],
    classes: set,
    split_dir: Path,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    dataset_config: EuroCropsDatasetConfigSSL,
    loss_fn: nn.Module,
    self_supervised: bool,
    presto_input: bool,
    masking_strategy: Literal["random", "fixed", None],
    max_samples: int | str,
    class_ids_to_names: dict[str, str] | None,
    max_masked_ratio: float | None = 0.7,
    model_channels: list[str] | None = None,
    pretrained: bool = False,
    ndvi_flag: bool = True,
    padding_value: float = 0.0,
) -> Task:
    """Load EuroCrops data.

    Args:
        mode: Whether to load pretrain or finetuning dataset.
        classes: The classes of the requested dataset split.
        split_dir: Directory where split is loaded from.
        preprocess_config: Config model of preprocessed data.
        dataset_config: Config model of dataset to be loaded.
        loss_fn: Loss function used to calculate the model's loss.
        self_supervised: Flag to make dataset usable in a self-supervised setting.
        presto_input: Whether it's a Presto input or not.
        masking_strategy: Which masking strategy to use. "random" splits the masked tokens randomly
            between the items in the batch. "fixed" is used for the implementation of the Presto
            channel encoding and will mask the same number of tokens in each batch item.
            None is just used for fine-tuning where we do not apply any masking at all.
        max_samples: Maximum number of samples per class within finetuning dataset.
        class_ids_to_names: Optional mapping from class identifiers to readable class names.
        max_masked_ratio: Maximum ratio of masked elements in data.
            Defaults to 0.7.
        model_channels: List of used data bands/channels (e.g. "VV", "VH2, "total_precipiation")
            in the network. Used for potential NDVI calculation, masking during fine-tuning
            and group masking during pre-training.
        pretrained: Flag whether the network was pre-trained or whether we fine-tune a random
            initialization.
        ndvi_flag: Whether to add NDVI if Sentinel-2 is available. Defaults to True.
        padding_value: Padding value used for padding data sequences.

    Returns:
        Task containg train, validation, and optionally test dataset.

    Raises:
        FileNotFoundError: If dataset split file is not found.
        FileNotFoundError: If train and/or validation set do not exist.
        FileNotFoundError: If finetuning mode and test set does not exist.
    """

    if mode == "finetuning" and model_channels is None and pretrained:
        raise AssertionError(
            "model_channels cannot be None when fine-tuning a pre-rained model. "
            "Please set them to the channels used during pre-training."
        )

    class_list = list(classes)  # ensure matching ordering between names and encoding
    if class_ids_to_names is not None:  # use readable class names if available
        class_names = [class_ids_to_names[str(c)] for c in class_list]
    else:  # use class identifiers as names otherwise
        class_names = [str(c) for c in class_list]
    encoding = {int(c): i for i, c in enumerate(class_list)}

    if mode == "finetuning":
        split_file = split_dir.joinpath(
            "finetune", f"{dataset_config.split}_split_{max_samples}.json"
        )
    else:
        split_file = split_dir.joinpath("pretrain", f"{dataset_config.split}_split.json")
    if split_file.exists():
        with open(split_file) as outfile:
            data_split = json.load(outfile)

    else:
        raise FileNotFoundError(
            str(split_file) + " does not exist. Please first build the dataset split."
        )

    satellites = dataset_config.data_sources

    satellites = [s for s in satellites if s in ["S1", "S2"]]
    satellites.sort()

    data_satellite_split: dict[str, dict[str, list[Path]]] = {
        key: {s: [] for s in satellites} for key in data_split
    }

    data_satellite_split = {
        key: {
            s: [
                preprocess_config.preprocess_dir.joinpath(s, str(dataset_config.year), file)
                for file in file_list
            ]
            for s in satellites
        }
        for key, file_list in data_split.items()
    }

    try:
        train = data_satellite_split["train"]
        train_list = reduce(add, train.values())
        val = data_satellite_split["val"]
        val_list = reduce(add, val.values())
        if mode == "finetuning":
            test = data_satellite_split["test"]
            test_list = [item for sublist in test.values() for item in sublist]
        else:
            test = None
            test_list = None
    except KeyError as err:
        raise FileNotFoundError() from err

    logger.info(f"Computing {mode} task.")
    era5_npz = Path(Settings().era5_data_dir) / "era5_npz"
    mmap_store = MMapStoreERA5(
        train_list + val_list + (test_list if test_list is not None else []), era5_npz
    )

    metrics = (
        []
        if self_supervised
        else get_metrics(
            dataset_config.metrics,
            num_classes=len(class_names),
            class_names=class_names,
        )
    )

    task = Task(
        task_id="eurocrops",
        train_set=TransformDataset(
            EuroCropsDatasetSSL(
                file_dict=train,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                preprocess_config=preprocess_config,
                self_supervised=self_supervised,
                model_channels=model_channels,
                presto_input=presto_input,
                pretrained=pretrained,
                ndvi_flag=ndvi_flag,
                padding_value=padding_value,
            ),
            collate_fn=partial(
                custom_collate_fn,
                max_masked_ratio=max_masked_ratio,
                model_channels=model_channels,
                padding_value_data=padding_value,
                masking_strategy=masking_strategy,
            ),
        ),
        val_set=TransformDataset(
            EuroCropsDatasetSSL(
                file_dict=val,
                encode=encoding,
                mmap_store=mmap_store,
                config=dataset_config,
                preprocess_config=preprocess_config,
                self_supervised=self_supervised,
                model_channels=model_channels,
                presto_input=presto_input,
                pretrained=pretrained,
                ndvi_flag=ndvi_flag,
                padding_value=padding_value,
            ),
            collate_fn=partial(
                custom_collate_fn,
                max_masked_ratio=max_masked_ratio,
                model_channels=model_channels,
                padding_value_data=padding_value,
                masking_strategy=masking_strategy,
            ),
        ),
        test_set=(
            TransformDataset(
                EuroCropsDatasetSSL(
                    file_dict=test,
                    encode=encoding,
                    mmap_store=mmap_store,
                    config=dataset_config,
                    preprocess_config=preprocess_config,
                    self_supervised=self_supervised,
                    model_channels=model_channels,
                    presto_input=presto_input,
                    pretrained=pretrained,
                    ndvi_flag=ndvi_flag,
                    padding_value=padding_value,
                ),
                collate_fn=partial(
                    custom_collate_fn,
                    max_masked_ratio=max_masked_ratio,
                    model_channels=model_channels,
                    padding_value_data=padding_value,
                    masking_strategy=masking_strategy,
                ),
            )
            if test
            else None
        ),
        num_classes=len(encoding.keys()),
        loss_fn=loss_fn,
        metrics=metrics,
    )

    return task
