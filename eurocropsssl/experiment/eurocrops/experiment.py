import logging
from typing import Literal

import eurocropsml.settings as eurocropsmlsettings
from eurocropsml.dataset.splits import get_split_dir

from eurocropsssl.dataset.task import Task
from eurocropsssl.experiment.eurocrops.config import EuroCropsTransferConfig
from eurocropsssl.experiment.eurocrops.utils import load_task as eurocrops_load_task
from eurocropsssl.experiment.sen12mscrts.utils import load_task as sen12mscrts_load_task
from eurocropsssl.experiment.transfer import (
    TransferExperiment,
    TransferExperimentBuilder,
    build_pretrain_experiment,
)
from eurocropsssl.settings import Settings

logger = logging.getLogger(__name__)


class EuroCropsExperimentBuilder(TransferExperimentBuilder[EuroCropsTransferConfig]):
    """Class for building EuroCrops experiments."""

    def __init__(self, config: EuroCropsTransferConfig):
        super().__init__(config)
        self.dataset_config = config.eurocrops_dataset

    def build_experiment(self, mode: Literal["pretrain", "finetune"]) -> TransferExperiment:
        """Build transfer experiment."""
        experiment_dir = Settings().experiment_dir
        experiment_dir.mkdir(exist_ok=True, parents=True)

        data_dir = eurocropsmlsettings.Settings().data_dir

        split_dir = get_split_dir(data_dir, self.config.split.base_name)

        Task.DATALOADER_NUM_WORKERS = self.config.pretrain.train_config.dataloader_num_workers

        if mode == "finetune":
            if self.config.pretrain_dataset is not None:
                pretrained: bool = True
            else:
                pretrained = False
            finetuning_tasks = {
                f"eurocrops_finetuning_maxsamples_{num}": (
                    eurocrops_load_task(
                        split_dir,
                        mode="finetuning",
                        split_config=self.config.split,
                        preprocess_config=self.config.preprocess,
                        dataset_config=self.dataset_config,
                        self_supervised=False,
                        presto_input=self.config.presto_input,
                        max_samples=num,
                        max_masked_ratio=None,
                        masking_strategy=None,
                        model_channels=self.config.model.channels,
                        pretrained=pretrained,
                        ndvi_flag=self.config.ndvi_flag,
                    ),
                    self.config.finetune,
                )
                for num in self.dataset_config.max_samples
            }
            return build_pretrain_experiment(
                pretrain_experiment_config=self.config.pretrain,
                pretrain_task=None,
                finetuning_tasks=finetuning_tasks,
                model_config=self.config.model,
                experiment_dir=experiment_dir,
                self_supervised=True,
            )
        else:
            if self.config.model.channel_encoding in ["presto", "cross_attn"]:
                # this will mask a fixed number of tokens and use Presto's channel encoding
                masking_strategy: Literal["random", "fixed"] = "fixed"
            else:
                # this will mask a random number of tokens and use our own channel encoding
                masking_strategy = "random"
            logger.info(f"Setting masking strategy to {masking_strategy}.")

            if self.config.pretrain_dataset is not None:
                pretrain_task = sen12mscrts_load_task(
                    self.config.pretrain_dataset,
                    max_masked_ratio=self.config.pretrain.train_config.max_masked_ratio,
                    masking_strategy=masking_strategy,
                    model_channels=self.config.model.channels,
                )
            else:
                pretrain_task = eurocrops_load_task(
                    split_dir,
                    mode="pretraining",
                    split_config=self.config.split,
                    preprocess_config=self.config.preprocess,
                    dataset_config=self.dataset_config,
                    self_supervised=True,
                    presto_input=self.config.presto_input,
                    max_masked_ratio=self.config.pretrain.train_config.max_masked_ratio,
                    masking_strategy=masking_strategy,
                    model_channels=self.config.model.channels,
                    ndvi_flag=self.config.ndvi_flag,
                )
            return build_pretrain_experiment(
                pretrain_experiment_config=self.config.pretrain,
                pretrain_task=pretrain_task,
                finetuning_tasks=None,
                model_config=self.config.model,
                experiment_dir=experiment_dir,
                self_supervised=True,
            )
