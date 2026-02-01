import logging
from typing import Literal, cast

from eurocropsml.dataset.config import EuroCropsDatasetConfig

from eurocropsssl.dataset.config import ERA5_BANDS

logger = logging.getLogger(__name__)


class EuroCropsDatasetConfigSSL(EuroCropsDatasetConfig):
    """EuroCrops dataset config for SSL, adding additional data sources."""

    data_sources: list[Literal["S1", "S2", "ERA5"]] = ["S2"]  # added ERA5
    era5_channels: list[Literal["2m_temperature", "total_precipitation"]] | None = None

    def post_init(self) -> None:
        """Custom post init logic for different data sources."""
        # run parent logic
        super().post_init()
        if "ERA5" in self.data_sources:
            if self.era5_channels is None:
                self.era5_channels = cast(list, ERA5_BANDS)
                logger.info(
                    "No ERA5 variables defined. Setting to defaults " f"{', '.join(ERA5_BANDS)}."
                )
            self.total_num_channels += len(self.era5_channels)
