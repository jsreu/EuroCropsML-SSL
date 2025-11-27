import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LocationEncoding(nn.Module):
    """Location Encoding for timeseries input vector.

    Args:
        d_hid: Input dimension for Location Encoding.
    """

    def __init__(self, d_hid: int):
        super().__init__()
        self.latlon_embed = nn.Linear(3, d_hid)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        """Convert latitude/longitude coordinates to 3D Cartesian coordinates.

        Args:
            latlons: Tensor of [latitude, longitude] values in degrees

        Returns:
            Tensor of [x, y, z] Cartesian coordinates
        """
        # copied from Presto
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, 0], latlon_radians[:, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Add location embeddings to input sequence.

        Args:
            x: Input sequence tensor
            centers: Tensor of geographic coordinates [latitude, longitude]

        Returns:
            Sequence tensor with location embedding prepended
        """
        embeded_centers = self.latlon_embed(self.cartesian(centers).to(torch.float))
        return torch.cat((embeded_centers.unsqueeze(-2), x), dim=1)
