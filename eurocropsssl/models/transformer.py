from typing import Any, Literal, cast

import torch
import torch.nn as nn
from eurocropsml.dataset.base import DataItem

from eurocropsssl.models.base import Model, ModelBuilder, ModelConfig
from eurocropsssl.models.channel_encoding import ChannelEncoding
from eurocropsssl.models.cross_attention import CrossAttentionBlock
from eurocropsssl.models.location_encoding import LocationEncoding
from eurocropsssl.models.positional_encoding import PositionalEncoding


class TransformerBackbone(nn.Module):
    """Transformer Backbone Encoder with multi-headed self-attention.

    Args:
        in_channels: Number of input channels.
        d_model: Number of input features.
            Input dimension of query, key, and value's linear layers.
        encoder_layer: Instance of TransformerEncoderLayer class.
            It consists of self-attention and a feedforward network.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        location_encoding: Whether to encode location of time series input vector.
        channel_encoding: Which channel encoding to use.
            "presto" for using the one from the presto implementation.
            "cross_attn" for using cross-attention in the decoder (CrossMAE).
                The channel encoding is the same as the one from Presto.
            "ours" for using our own implementation.
            None for using no channel encoding at all.
        channels: List of used data bands/channels (e.g. S1, S2, ERA5).
            Can be None if channel_encoding is False.
        t: Period to use for positional encoding.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        encoder_layer: nn.TransformerEncoderLayer,
        num_layers: int,
        pos_enc_len: int,
        location_encoding: bool,
        channel_encoding: Literal["presto", "ours", "cross_attn", None],
        channels: list,
        t: int = 1000,
    ):
        super().__init__()
        if channel_encoding is not None and channels is None:
            raise AssertionError("If channel encoding is used, channels cannot be None.")
        self._in_layernorm = nn.LayerNorm(in_channels)
        pos_enc_size = d_model
        if location_encoding:
            self._loc_encoding: LocationEncoding = LocationEncoding(d_model)
        if channel_encoding in ["presto", "cross_attn"]:
            chan_enc_len = int(d_model * 0.25)
            self._channel_encoding = ChannelEncoding(
                d_channel=chan_enc_len,
                embedding_size=d_model,
                channels=cast(list, channels),
            )
            # embedding consists of channel and positional embedding
            pos_enc_size -= chan_enc_len
        elif channel_encoding == "ours":
            raise NotImplementedError
        self._pos_encoding = PositionalEncoding(pos_enc_size, pos_enc_len, t)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self._in_linear = torch.nn.Linear(in_channels, pos_enc_size)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute model (backbone) forward pass."""
        x = x.float()
        x = self._in_layernorm(x)
        # get input x before applying linear transformation
        chan_x = x
        # apply linear layer
        x = self._in_linear(x)
        mask = kwargs.get("mask")
        if hasattr(self, "_channel_encoding"):
            if mask is None:
                mask = torch.zeros_like(chan_x, device=x.device).float()
            # apply positional encoding
            # pos_encoding: [B, T, d_model*0.75]
            pos_encoding: torch.Tensor = self._pos_encoding(x, kwargs.get("dates"), add_to_x=False)
            enc_output, original_indices, mask = self._channel_encoding(chan_x, pos_encoding, mask)
        else:
            if mask is None:
                mask = torch.zeros_like(x, device=x.device).float()
            # apply positional encoding
            # pos_encoding: [B, T, d_model]
            enc_output = self._pos_encoding(x, kwargs.get("dates"))
            # mask full time step if all channels are masked
            # transformer works on sequence level, so individual padded channels are ignored
            # by the network as long as the padding_value_data is not a valid data value
            if mask.dim() == enc_output.dim():
                mask = mask.all(dim=-1)
            original_indices = None
        if hasattr(self, "_loc_encoding"):
            # concatenates loc_encoding along -2 (append as timestep)
            enc_output = self._loc_encoding(enc_output, kwargs.get("center"))
            # add dimension to mask
            mask = torch.cat((torch.zeros(x.shape[0])[:, None].to(x.device), mask), dim=1)
        encoder_out = self._encoder(enc_output, src_key_padding_mask=mask)

        return cast(torch.Tensor, encoder_out), mask, original_indices


class TransformerSelfSupervisedHead(nn.Module):
    """Transformer Decoder Head for Self-supervision.

    Args:
        in_dim: Number of input dimentions.
        out_channels: Number of output channels.
        encoder_layer: Instance of TransformerEncoderLayer class.
            It consists of self-attention and a feedforward network.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        location_encoding: Whether to encode location of time series input vector.
        channel_encoding: Which channel encoding to use.
            "presto" for using the one from the presto implementation.
            "cross_attn" for using cross-attention in the decoder (CrossMAE).
                The channel encoding is the same as the one from Presto.
            "ours" for using our own implementation.
            None for using no channel encoding at all.
        channels: List of used data bands/channels (e.g. S1, S2, ERA5).
            Can be None if channel_encoding is False.
        t: Period to use for positional encoding.
    """

    def __init__(
        self,
        n_heads: int,
        in_dim: int,
        dim_fc: int,
        out_channels: int,
        encoder_layer: nn.TransformerEncoderLayer,
        num_layers: int,
        pos_enc_len: int,
        location_encoding: bool,
        channel_encoding: Literal["presto", "ours", "cross_attn", None],
        channels: list,
        t: int = 1000,
    ):
        super().__init__()
        self.pos_enc_size = in_dim
        self.channels = channels
        self.location_encoding = location_encoding
        if location_encoding:
            self._loc_encoding: LocationEncoding = LocationEncoding(in_dim)
        self.channel_encoding = channel_encoding
        if channel_encoding in ["presto", "cross_attn"]:
            self.chan_enc_len = int(in_dim * 0.25)
            self._channel_encoding = ChannelEncoding(
                d_channel=self.chan_enc_len,
                embedding_size=in_dim,
                channels=cast(list, channels),
            )
            # embedding consists of channel and positional embedding
            self.pos_enc_size -= self.chan_enc_len
            if channel_encoding == "cross_attn":
                self.cross_attention_decoder_blocks = nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            in_dim,
                            in_dim,
                            n_heads,
                            dim_fc,
                            qkv_bias=True,
                            qk_scale=None,
                            norm_layer=nn.LayerNorm,
                        )
                        for _ in range(num_layers)
                    ]
                )
        elif channel_encoding == "ours":
            raise NotImplementedError
        self._pos_encoding = PositionalEncoding(self.pos_enc_size, pos_enc_len, t)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self._in_linear_decoder = torch.nn.Linear(in_dim, in_dim)
        self._out_linear = nn.Linear(in_dim, out_channels)

    def forward(
        self,
        z: torch.Tensor,
        indices: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute model decoder forward pass."""

        z = z.float()
        z = self._in_linear_decoder(z)  # decoder embeddings
        mask = enc_mask if enc_mask is not None else cast(torch.Tensor, kwargs.get("mask"))
        if mask is not None and mask.dim() == z.dim():
            # if z (B, L, D) == mask (B, T, C)
            # => mask transformed to (B, T)
            mask = mask.all(dim=-1)

        if self.location_encoding:
            z = z[:, 1:, :]  # remove locations from timesteps
            if mask is not None:
                mask = mask[:, 1:]

        if self.channel_encoding == "presto":
            z, _ = self._channel_encoding.add_masked_tokens(z, indices, cast(torch.Tensor, mask))
            num_channel_groups = len(self._channel_encoding.band_group_to_idx)
            num_timesteps = int(z.shape[1] / num_channel_groups)

            # generat base positional encodings for original timesteps to later replicate
            pos_encoding = self._pos_encoding(
                z[:, :num_timesteps, : self.pos_enc_size],
                kwargs.get("dates"),
                add_to_x=False,
            )
            z_embeddings = self._channel_encoding.add_embeddings(
                z, pos_encoding, num_timesteps, num_channel_groups
            )
            decoder_out = self._encoder(z_embeddings)
            x = self._channel_encoding.reconstruct_inputs(decoder_out)
        elif self.channel_encoding == "cross_attn":
            # full_z are all tokens. We still need z (the encoder's output without masked token)
            # as the key in cross-attention
            full_z, full_mask = self._channel_encoding.add_masked_tokens(
                z, indices, cast(torch.Tensor, mask)
            )
            num_channel_groups = len(self._channel_encoding.band_group_to_idx)
            num_timesteps = full_z.shape[1] // num_channel_groups
            batch_size, _, embed_dim = full_z.shape

            # generate base positional encodings for original timesteps to later replicate
            pos_encoding = self._pos_encoding(
                full_z[:, :num_timesteps, : self.pos_enc_size],
                kwargs.get("dates"),
                add_to_x=False,
            )
            z_embeddings = self._channel_encoding.add_embeddings(
                full_z, pos_encoding, num_timesteps, num_channel_groups
            )
            # get masked tokens
            mask_bool = full_mask.bool()
            num_masked_tokens_per_sample = full_mask.sum(dim=1)  # Shape: [batch_size]
            max_masked_tokens = int(num_masked_tokens_per_sample.max().item())
            masked_tokens = z_embeddings[
                mask_bool
            ]  # Shape: [total masked tokens in batch, embed_dim]

            # Create padded tensor and valid indices
            masked_tokens_padded = torch.zeros(
                batch_size, max_masked_tokens, embed_dim, device=z_embeddings.device
            )
            valid_indices = torch.arange(max_masked_tokens, device=z_embeddings.device).unsqueeze(
                0
            ) < num_masked_tokens_per_sample.unsqueeze(1)

            # scatter masked indices
            batch_indices = torch.arange(
                batch_size, device=z_embeddings.device, dtype=torch.long
            ).repeat_interleave(num_masked_tokens_per_sample.long())

            token_indices = torch.cat(
                [
                    torch.arange(count, device=z_embeddings.device)
                    for count in num_masked_tokens_per_sample.long()
                ]
            )

            # place masked tokens into padded tensor using valid indices
            masked_tokens_padded[batch_indices, token_indices] = masked_tokens

            # K, V are all from z (encoder output)
            # Q are the masked tokens
            for _, blk in enumerate(self.cross_attention_decoder_blocks):
                masked_tokens_padded = blk(
                    masked_tokens_padded, z
                )  # z is encoder's output without masked tokens
            all_tokens = z_embeddings.clone()  # Start with the original tensor

            all_tokens[full_mask.bool()] = masked_tokens_padded[valid_indices].view(-1, embed_dim)
            x = self._channel_encoding.reconstruct_inputs(all_tokens)
        else:
            if mask is not None:
                z = torch.masked_fill(z, mask.unsqueeze(-1).bool(), 0)
            z = self._pos_encoding(z, kwargs.get("dates"))
            decoder_out = self._encoder(z)
            x = self._out_linear(decoder_out)

        # we want masked only loss => return only masked items
        if (aug_mask := kwargs.get("aug_mask")) is not None:
            x = x[aug_mask]
        return cast(torch.Tensor, x)


class TransformerConfig(ModelConfig):
    """Config for transformer model.

    Args:
        n_heads: Number of heads in the multi-head attention models.
        in_channels: Number of input channels.
        d_model: Number of input features. Input dimension of query, key, and value's linear layers.
        dim_fc: Dimensionality of query, key, and value used as input to the multi-head attention.
        num_layers: Number of sub-encoder-layers in the encoder.
        pos_enc_len: Length of positional encoder table.
        location_encoding: Whether to encode location of time series input vector.
        channel_encoding: Which channel encoding to use.
            "presto" for using the one from the presto implementation.
            "ours" for using our own implementation.
            None for using no channel encoding at all.
        channels: List of used data bands/channels. Can be None if channel_encoding is False.
    """

    n_heads: int
    in_channels: int
    d_model: int
    dim_fc: int
    num_layers: int
    pos_enc_len: int
    location_encoding: bool = False
    channel_encoding: Literal["presto", "cross_attn", "ours", None] = None
    channels: list[str] | None = None

    model_builder: Literal["TransformerModelBuilder"] = "TransformerModelBuilder"


class TransformerModel(Model):
    """Model architecture for pretraining and finetuning with transformers."""

    def forward(self, ipt: DataItem) -> torch.Tensor:
        """Compute forward pass for Transformer model."""
        out, mask, indices = self.backbone(ipt.data, **ipt.meta_data)
        match self.head:
            case nn.Linear():
                if mask is not None:
                    # make sure mask has type float
                    mask = mask.float()
                    # set masked tokens to 0
                    out = out * (1 - mask.unsqueeze(-1))
                    out = out.sum(dim=1) / torch.sum(1 - mask, -1, keepdim=True)
                    return cast(torch.Tensor, self.head(out))
                else:
                    return cast(torch.Tensor, self.head(out.mean(1)))
            case TransformerSelfSupervisedHead():
                return cast(
                    torch.Tensor,
                    self.head(out, cast(torch.Tensor, indices), enc_mask=mask, **ipt.meta_data),
                )
            case _:
                raise NotImplementedError


class TransformerModelBuilder(ModelBuilder):
    """Transformer with multi-headed self-attention.

    Args:
        config: Transformer model config.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_fc,
            batch_first=True,
        )

    def build_backbone(self) -> TransformerBackbone:
        """Build backbone for Transformer model."""
        return TransformerBackbone(
            in_channels=self.config.in_channels,
            d_model=self.config.d_model,
            encoder_layer=self._encoder_layer,
            num_layers=self.config.num_layers,
            pos_enc_len=self.config.pos_enc_len,
            location_encoding=self.config.location_encoding,
            channel_encoding=self.config.channel_encoding,
            channels=self.config.channels,
        )

    def build_classification_head(self, num_classes: int) -> nn.Linear:
        """Build linear classification head for Transformer model."""
        return nn.Linear(self.config.d_model, num_classes)

    def build_self_supervised_head(self) -> nn.Module:
        """Build SSL decoder for Transformer model."""
        return TransformerSelfSupervisedHead(
            n_heads=self.config.n_heads,
            in_dim=self.config.d_model,
            dim_fc=self.config.dim_fc,
            out_channels=self.config.in_channels,
            encoder_layer=self._encoder_layer,
            num_layers=self.config.num_layers,
            pos_enc_len=self.config.pos_enc_len,
            location_encoding=self.config.location_encoding,
            channel_encoding=self.config.channel_encoding,
            channels=self.config.channels,
        )

    def build_classification_model(self, num_classes: int, device: torch.device) -> Model:
        """Build classification Transformer model."""
        backbone = self.build_backbone()
        head = self.build_classification_head(num_classes)
        return TransformerModel(backbone=backbone, head=head, device=device)

    def build_self_supervised_model(self, device: torch.device) -> Model:
        """Build SSL Autoencoder model."""
        backbone = self.build_backbone()
        head = self.build_self_supervised_head()
        return TransformerModel(backbone=backbone, head=head, device=device)
