import logging
from typing import Any

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig, EuroCropsSplit
from pydantic import BaseModel

from eurocropsssl.dataset.eurocrops.config import EuroCropsDatasetConfigSSL
from eurocropsssl.dataset.sen12mscrts.config import SEN12MSCRTSDatasetConfig
from eurocropsssl.experiment.base import TrainExperimentConfig
from eurocropsssl.models.presto import PrestoConfig
from eurocropsssl.models.transformer import TransformerConfig

logger = logging.getLogger(__name__)


class EuroCropsTransferConfig(BaseModel):
    """Configuration for the EuroCrops transfer experiment.

    Args:
        base_name: Name experiments are stored under.
        model: Configuration for (pre)-trained model.
        pretrain: Experiment configuration for pretraining.
        finetune: Experiment configuration for finetuning experiment.
        eurocrops_dataset: Configuration for EuroCrops dataset.
        split: Configuration for EuroCrops splits.
        preprocess: Configuration for preprocessing EuroCrops dataset.
        presto_input: Flag to make data usable for self-supervied learning with the Presto model.
            initialization.
        ndvi_flag: Whether to add NDVI if Sentinel-2 is available. Defaults to True.
        pretrain_dataset: Config for pre-train dataset. Currently only SEN12MSCRTS is implemented.
            If None and presto_input is True, it will fine-tune the Presto backbone.
            Otherwise, it will be pre-trained on EuroCrops. Please note that this also needs to be
            set if fine-tuning a pre-trained network (except for Presto).
    """

    base_name: str
    model: TransformerConfig | PrestoConfig
    pretrain: TrainExperimentConfig
    finetune: TrainExperimentConfig

    eurocrops_dataset: EuroCropsDatasetConfigSSL

    split: EuroCropsSplit
    preprocess: EuroCropsDatasetPreprocessConfig

    presto_input: bool = False

    ndvi_flag: bool = True

    pretrain_dataset: SEN12MSCRTSDatasetConfig | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.post_init()

    def post_init(self) -> None:
        """Make dynamic config based on initialized params."""
        # default classification metrics can't be used during self-supervised pretraining
        self.pretrain.key_metric = "loss"
        if hasattr(self.model, "location_encoding"):
            self.model.location_encoding = True
        if isinstance(self.model, PrestoConfig):
            # Presto model is a self supervise model
            # Presto requires a special input
            self.presto_input = True
            if self.pretrain_dataset is not None:
                self.pretrain_dataset = None
                logger.info(
                    "The presto model cannot be pre-trained with SEN12MS-CR-TS data. Setting "
                    "EuroCropsTransferConfig.pretrain_dataset to None."
                )

        if self.presto_input:
            self.ndvi_flag = False  # is calculated within Presto
            # Presto construct input already normalizes values
            # Presto only works for monthly temporal frequency
            self.eurocrops_dataset.date_type = "month"
            # Presto expects unnormalized inputs and will take care of its own normalization
            # see utils.py and pipelines/s1_s2_era5_srtm.py at
            # https://github.com/nasaharvest/presto/blob/main/presto/dataops/
            self.eurocrops_dataset.normalize = False
            self.eurocrops_dataset.remove_s2_bands = ["01", "09", "10"]
            if self.pretrain_dataset is not None:
                self.pretrain_dataset = None
                logger.info(
                    "The presto model cannot be pre-trained with SEN12MS-CR-TS data. Setting "
                    "EuroCropsTransferConfig.pretrain_dataset to None."
                )

        if self.eurocrops_dataset.date_type == "month" and hasattr(self.model, "pos_enc_len"):
            # Make sure positional encoder works with temporal frequency of data
            self.model.pos_enc_len = 12
