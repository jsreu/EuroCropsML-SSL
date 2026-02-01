from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, cast

import torch
from eurocropsml.dataset.base import LabelledData
from torch.nn.utils.rnn import pad_sequence

from eurocropsssl.dataset.config import _build_band_groups

logger = logging.getLogger(__name__)


def _get_mask(
    tensor_data: torch.Tensor,
    max_masked_ratio: float = 0.7,
    model_channels: list[str] | None = None,
) -> torch.Tensor | None:

    if max_masked_ratio <= 0:
        return None

    # Get dimensions of tensor_data (batch, timesteps, channels)
    batch_size, num_timesteps, num_channels = tensor_data.shape

    # Maximum number of elements that can be masked in a single tensor
    elements_per_sample = num_timesteps * num_channels
    num_masked_elements_per_sample = int(elements_per_sample * max_masked_ratio)

    # Select a masking strategy
    strategies = [
        "time_contiguous",
        "time_random",
        "random",
    ]
    if model_channels is not None:
        strategies.append("group_channels")
    else:
        logger.warning(
            "If you want to mask channels (groups), "
            "please indicate the available channels in the model config."
        )
    masking_strategy = random.choice(strategies)
    if masking_strategy == "random":
        # Generate a mask of the same shape as tensor_data
        mask = torch.zeros(tensor_data.shape, device=tensor_data.device)

        # Generate random indices for each batch element
        # Shape: [batch_size, num_masked_elements_per_sample]
        flat_indices = torch.argsort(
            torch.rand(batch_size, elements_per_sample, device=tensor_data.device),
            dim=1,
        )[:, :num_masked_elements_per_sample]

        # Convert to advanced indexing format
        batch_indices = (
            torch.arange(batch_size, device=tensor_data.device)
            .unsqueeze(1)
            .expand(-1, num_masked_elements_per_sample)
        )

        # Set the selected indices to True
        mask.view(batch_size, -1)[batch_indices, flat_indices] = True
    elif masking_strategy == "group_channels":
        # setup
        length_divisor = num_timesteps
        max_maskable_length = num_masked_elements_per_sample // length_divisor
        mask = torch.zeros_like(tensor_data)
        # band setup
        band_groups = _build_band_groups(cast(list, model_channels))
        channels_per_group = {group: len(channels) for group, channels in band_groups.items()}
        key_to_index = {key: idx for idx, key in enumerate(band_groups.keys())}

        # group sampling
        available_groups = list(band_groups.keys())
        sampled_groups = []
        remaining_channels = max_maskable_length

        while remaining_channels > 0 and available_groups:
            group = random.choice(available_groups)
            num_channels = channels_per_group[group]

            if num_channels <= remaining_channels:
                sampled_groups.append(group)
                remaining_channels -= num_channels
                available_groups.remove(group)
            else:
                # group has too many channels, try another
                available_groups.remove(group)

        band_groups_to_mask = [key_to_index[key] for key in sampled_groups]
        mask[:, band_groups_to_mask] = True

    else:
        masking_type = masking_strategy.split("_")[-1]

        # Determine dimension parameters
        length_divisor = num_channels
        max_length = num_timesteps
        mask = torch.zeros_like(tensor_data)

        max_maskable_length = num_masked_elements_per_sample // length_divisor
        if masking_type == "contiguous":
            # Generate different random starting positions for each batch item
            init_pos = torch.randint(
                0,
                max_length - max_maskable_length + 1,
                (batch_size,),
                device=tensor_data.device,
            )

            # Create batch indices for advanced indexing
            batch_indices = torch.arange(batch_size, device=tensor_data.device).unsqueeze(1)

            # Create position indices for each batch
            indices = torch.stack(
                [
                    torch.arange(pos, pos + max_maskable_length, device=tensor_data.device)
                    for pos in init_pos
                ]
            )
            # Expand indices for broadcasting
            batch_indices_exp = batch_indices.expand(-1, max_maskable_length)

            # Expand channel indices across batch and time for broadcasting
            channel_indices_expanded = torch.arange(num_channels, device=tensor_data.device).expand(
                batch_size, max_maskable_length, -1
            )

            # Set mask for all channels at selected time positions
            mask[
                batch_indices_exp.unsqueeze(-1),
                indices.unsqueeze(-1),
                channel_indices_expanded,
            ] = True

        elif masking_type == "random":
            # uniformly sample random values
            rand_values = torch.rand((batch_size, max_length), device=tensor_data.device)
            # get indices from noise
            _, indices = torch.topk(rand_values, max_maskable_length, dim=1)

            # Create and expand batch indices
            batch_indices_exp = (
                torch.arange(batch_size, device=tensor_data.device)
                .unsqueeze(1)
                .expand(-1, max_maskable_length)
            )

            # Expand channel indices across batch and time for broadcasting
            channel_indices_expanded = torch.arange(num_channels, device=tensor_data.device).expand(
                batch_size, max_maskable_length, -1
            )
            # Set mask for all channels at selected time positions
            mask[
                batch_indices_exp.unsqueeze(-1),
                indices.unsqueeze(-1),
                channel_indices_expanded,
            ] = True

        else:
            raise NotImplementedError
    return mask


def custom_collate_fn(
    batch: Sequence[LabelledData],
    model_channels: list[str] | None = None,
    padding_value_data: float = 0.0,
    padding_value_no_data: int = -1,
    max_masked_ratio: float | None = None,
    masking_strategy: Literal["random", "fixed", None] = None,
) -> LabelledData:
    """Collate function for batch creation within data loader.

    Used to create batches from a dataset's DataItem.

    Args:
        batch: List of DataItem from dataset.
        model_channels: List of used data bands/channels (e.g. S1, S2, ERA5) in the network.
            Used for group masking. If None, group masking is enabled.
        padding_value_data: Value used for padding the data series.
        padding_value_no_data: Value used for padding tensors except for data.
        max_masked_ratio: Maximum number of tokens masked within a batch.
        masking_strategy: Which masking strategy to use. "random" splits the masked tokens randomly
            between the items in the batch. "fixed" is used for the implementation of the Presto
            channel encoding and will mask the same number of tokens in each batch item.
            None is just used for fine-tuning.

    Returns:
        New DataItem with batched data.
    """
    batch_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    tensor_stackability: dict[str, bool] = defaultdict(lambda: True)
    for item in batch:
        for tensor_name, tensor in item.to_tensor_dict().items():
            if tensor_stackability[tensor_name] and bool(
                prev_tensors := batch_tensors[tensor_name]
            ):
                tensor_stackability[tensor_name] = prev_tensors[-1].shape == tensor.shape
            batch_tensors[tensor_name].append(tensor)

    batched_tensors = {
        tensor_name: (
            torch.stack(tensors)
            if tensor_stackability[tensor_name]
            else pad_sequence(
                tensors,
                batch_first=True,
                padding_value=(
                    padding_value_data if tensor_name == "data" else padding_value_no_data
                ),
            )
        )
        for tensor_name, tensors in batch_tensors.items()
    }

    if (
        not tensor_stackability["label"]
        and batched_tensors["label"].shape != batched_tensors["data"].shape
    ):
        batched_tensors["label"] = torch.concat(batch_tensors["label"], 0)

    # create pad_mask based on padding value
    pad_mask = batched_tensors["data"].eq(padding_value_data)  # [B, T, C]

    # handle presto_mask case
    aug_mask = batched_tensors.get("presto_mask")
    if aug_mask is not None:
        # no need for pad_mask, invalid data is already masked when constructing the presto input
        batched_tensors["presto_mask"] = aug_mask.bool()

    # handle SSL masking case
    elif max_masked_ratio is not None:
        # set masking strategy
        if masking_strategy == "random":
            raise NotImplementedError  # TODO: Add once our own channel encoding is implemented
        elif masking_strategy == "fixed":
            masking_func = _get_mask
        elif masking_strategy is None:
            raise AssertionError("Please set a masking strategy or set max_masked_ratio to None.")

        # generate mask and apply it
        aug_mask = masking_func(batched_tensors["data"], max_masked_ratio, model_channels)
        if aug_mask is not None:  # [B, T, C]
            # update labels based on mask by taking only the masked pixels
            batched_tensors["label"] = batched_tensors["label"][aug_mask.bool()]

            # combine masks with dimension handling
            if aug_mask.dim() != pad_mask.dim():
                # remove channel dimension from pad_mask
                # individual padded channels are ignored by the network as long as the
                # padding_value_data is not a valid data value
                batched_tensors["mask"] = torch.logical_or(
                    aug_mask.bool(), pad_mask.all(-1)  # [B, T]
                )
            else:
                # keep all dimensions
                batched_tensors["mask"] = torch.logical_or(
                    aug_mask.bool(), pad_mask  # [B, T, C]  # [B, T, C]
                )

            batched_tensors["aug_mask"] = aug_mask.bool()

    # handle fine-tuning case (EuroCrops dataset)
    elif (aug_mask := batched_tensors.get("aug_mask")) is not None:
        # only during fine-tuning, we receive a mask from the EuroCrops dataset class
        # aug_mask has shape of data tensor: [B, T, C]
        batched_tensors["mask"] = torch.logical_or(aug_mask.bool(), pad_mask)

    # default case: just use padding mask
    else:
        batched_tensors["mask"] = pad_mask

    return LabelledData.from_tensor_dict(batched_tensors)
