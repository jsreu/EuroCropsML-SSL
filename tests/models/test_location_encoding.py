import numpy as np
import pytest
import torch
from eurocropsml.dataset.base import DataItem

from eurocropsssl.models.location_encoding import LocationEncoding


@pytest.fixture
def location_encoder() -> LocationEncoding:
    return LocationEncoding(d_hid=128)


@pytest.fixture
def linear_layer() -> torch.nn.Linear:
    return torch.nn.Linear(13, 128)


@pytest.fixture
def test_data_item() -> DataItem:
    # create random data for time series
    data = np.ones((100, 13))
    # padding for batching
    data = np.concatenate((data, -1 * np.ones((10, 13))), axis=0)  # type: ignore[assignment]
    tensor_data = torch.tensor(data, dtype=torch.float)

    # random coordinates for testing
    # format: [latitude, longitude]
    centers = np.random.uniform(-90, 90, (1, 2))  # Random lat/lon

    return DataItem(
        data=tensor_data.unsqueeze(0),
        meta_data={"centers": torch.tensor(centers, dtype=torch.float)},
    )


def test_location_encoding(
    location_encoder: LocationEncoding,
    test_data_item: DataItem,
    linear_layer: torch.nn.Linear,
) -> None:
    # Get data and location centers
    data = test_data_item.data
    centers = test_data_item.meta_data["centers"]

    # Process the raw data through a linear layer first
    data = linear_layer(data)
    batch_size, seq_len, hidden_dim = data.size()

    # Apply location encoding
    new_data = location_encoder(data, centers)

    # verify that shape is as expected (one additional timestep for the location encoding)
    assert new_data.size(0) == batch_size
    assert new_data.size(1) == seq_len + 1  # one extra timestep for location encoding
    assert new_data.size(2) == hidden_dim

    # calculate expected cartesian coordinates for verification
    cartesian_coords = location_encoder.cartesian(centers)
    assert cartesian_coords.size() == (centers.size(0), 3)

    # get first timestep which should be location encoding
    location_embed = new_data[:, 0, :]

    # verify location encoding is not all zeros
    assert not torch.allclose(location_embed, torch.zeros_like(location_embed))

    # check that the original data is preserved in the remaining timesteps
    assert torch.allclose(new_data[:, 1:, :], data)


def test_cartesian_conversion() -> None:
    """Test that the cartesian conversion works correctly."""
    # test with known values
    latlons = torch.tensor([[0.0, 0.0, 0.0]])  # Equator at Greenwich
    cartesian = LocationEncoding.cartesian(latlons)

    # at (0,0,0), cartesian should be (1,0,0)
    assert torch.allclose(cartesian, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)

    # test with North Pole
    latlons = torch.tensor([[90.0, 0.0, 0.0]])
    cartesian = LocationEncoding.cartesian(latlons)

    # at north pole, cartesian should be (0,0,1)
    assert torch.allclose(cartesian, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)
