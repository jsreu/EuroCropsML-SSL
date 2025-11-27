import torch
import torch.nn as nn

from eurocropsssl.dataset.config import ALL_BANDS, _build_band_groups


# Copied and adapted from Presto
class ChannelEncoding(nn.Module):
    """Channel Encoding for timeseries input vector.

    Handles embedding of different spectral bands grouped by sensor type,
    with support for masking and reconstruction of channel data.

    Args:
        d_channel: Input dimension for Channel Encoding.
    """

    def __init__(self, d_channel: int, embedding_size: int = 128, channels: list[str] = ALL_BANDS):
        """Initialize channel encoding module.

        Args:
            d_channel: Dimension for channel embeddings
            embedding_size: Dimension of token embeddings
            channels: List of channel/band names to process
        """
        super().__init__()

        self.channels = channels

        # Create the OrderedDict
        self.band_groups = _build_band_groups(channels)

        self.band_group_to_idx = {name: idx for idx, name in enumerate(self.band_groups)}

        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )

        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups),
            embedding_dim=d_channel,
        )

        self.channel_decode = nn.ModuleDict(
            {
                group_name: nn.Linear(embedding_size, len(group))
                for group_name, group in self.band_groups.items()
            }
        )

        self.d_channel = d_channel

        self.mask_token = nn.Parameter(torch.zeros(embedding_size))

    def mask_tokens(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking and efficiently pack non-masked tokens.

        Args:
            x: Input tensor
            mask: Boolean mask where True indicates tokens to be masked

        Returns:
            Tuple of (packed tensor with masked tokens moved to end,
                        indices for restoring original positions,
                        updated mask for packed tensor)
        """
        # move all non-masked values to the front of their rows
        mask = mask.bool()
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask

    def add_masked_tokens(
        self, x: torch.Tensor, indices: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace masked tokens with mask embeddings and restore original sequence order.

        Args:
            x: Input tensor of unmasked tokens
            indices: Original indices for token positions
            x_mask: Mask indicating which tokens were masked

        Returns:
            Tuple of (tensor with mask tokens inserted at masked positions,
                    updated mask)
        """
        all_tokens = self.mask_token.expand(x.shape[0], indices.shape[1], -1)
        mask = torch.cat(
            (
                x_mask,
                torch.ones((x.shape[0], indices.shape[1] - x.shape[1]), device=x.device),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = all_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[~mask.bool()] = x[~x_mask.bool()]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)

        return (
            out,
            mask,
        )  # Shape out: [batch_size, timestep*num_channel_groups, embedding_size]

    def add_embeddings(
        self,
        x: torch.Tensor,
        positional_embedding: torch.Tensor,
        num_timesteps: int,
        num_channel_groups: int,
    ) -> torch.Tensor:
        """Add positional and channel embeddings to input tokens.

        Args:
            x: Input tensor
            positional_embedding: Positional embeddings tensor
            num_timesteps: Number of time steps in sequence
            num_channel_groups: Number of channel groups

        Returns:
            Tensor with positional and channel embeddings added
        """
        # repeat positional embeddings for each channel group to match time dimension
        positional_embedding = (
            positional_embedding.unsqueeze(2)
            .expand(-1, -1, num_channel_groups, -1)
            .reshape(positional_embedding.shape[0], -1, positional_embedding.shape[2])
        )

        # repeat channel embeddings (channel) for each time step to match time dimension
        channel_embeddings = torch.repeat_interleave(
            self.channel_embed.weight, repeats=num_timesteps, dim=0
        )

        # first add channel embeddings to x
        x[:, :, : channel_embeddings.size(-1)] += channel_embeddings
        # then add positional embeddings
        x[:, :, channel_embeddings.size(-1) :] += positional_embedding

        return x

    def reconstruct_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct original channel values from encoded representation.

        Args:
            x: Encoded tensor

        Returns:
            Reconstructed channel values
        """
        num_channel_groups = len(self.band_group_to_idx)
        num_timesteps = int(x.shape[1] / num_channel_groups)
        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        eo_output = [
            self.channel_decode[group_name](x[:, idx])
            for group_name, idx in self.band_group_to_idx.items()
        ]
        return torch.cat(eo_output, dim=-1)

    def forward(
        self, x: torch.Tensor, pos_encoding: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through channel encoding module.

        Args:
            x: Input tensor with channel data
            pos_encoding: Positional encoding tensor
            mask: Mask indicating which values to mask

        Returns:
            Tuple of (masked and encoded tensor,
                        original indices for reconstruction,
                        updated mask)
        """
        # assumption: number of masked patches is the same for all items in the batch.
        all_tokens, all_masks = [], []

        for channel_group, channel_idxs in self.band_groups.items():
            # get group data
            group_data = x[:, :, channel_idxs]  # [batch, timesteps, channels_in_group]
            # apply linear transformation to the channel group to encode channel info
            group_tokens = self.eo_patch_embed[channel_group](
                group_data
            )  # [batch, timesteps, embedding_dim]

            # get learned channel embedding for group ([d_model*0.25])
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(x.device)
            )
            # first add channel embeddings to group tokens
            group_tokens[:, :, : channel_embedding.size(-1)] += channel_embedding
            # then add positional embeddings
            group_tokens[:, :, channel_embedding.size(-1) :] += pos_encoding

            all_tokens.append(group_tokens)
            # mask entire token if any channel in group is masked
            # this also masks padded channels (e.g. if S1 is not available)
            # [batch, timesteps]
            group_mask = torch.max(mask[:, :, channel_idxs], dim=-1)[0]
            all_masks.append(group_mask)

        x = torch.cat(all_tokens, dim=1)  # [batch, timesteps*num_band_groups, embedding_dim]
        mask = torch.cat(all_masks, dim=1)  # [batch, timesteps*num_band_groups]

        # reorder and cut off masked tokens (lightweight version)
        x, orig_indices, upd_mask = self.mask_tokens(x, mask)

        return x, orig_indices, upd_mask.bool()
