# taken from https://github.com/TonyLianLong/CrossMAE/blob/main/transformer_utils.py
# slightly adjusted by Joana Reuss (TUM)
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """MLP as used in Vision Transformer and CrossMAE."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after MLP layers
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross attention block that enables interaction between encoder and decoder features.

    Implements a transformer block with cross-attention mechanism followed by MLP,
    allowing decoder features to attend to encoder outputs.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int,
        dim_fc: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(
            encoder_dim,
            decoder_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(decoder_dim)
        self.mlp = MLP(
            in_features=decoder_dim,
            hidden_features=dim_fc,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Forward pass of Cross Attention Block.

        Args:
            x: decoder feature
            encoder_output: encoder feature (after layernorm): key and value ´
        """
        x = x + self.cross_attn(self.norm1(x), encoder_output)
        x = x + self.mlp(self.norm2(x))

        return x


class CrossAttention(nn.Module):
    """Cross-attention mechanism between encoder and decoder features.

    Implements multi-head attention where queries come from decoder
    and keys/values come from encoder.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE: scale factor was wrong in my original version
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.linear_proj = nn.Linear(decoder_dim, decoder_dim)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Cross Attention forward pass.

        With query from the decoder, key and value from the encoder (enc_output).

        """

        B, T, C = x.shape
        T_encoder_output = encoder_output.shape[1]
        # head_dim = embedding_size (C) // num_heads
        q = (
            self.q(x).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        )  # Shape [B, num_heads, T, head_dim]
        kv = (
            self.kv(encoder_output)
            .reshape(B, T_encoder_output, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # Shape [B, num_heads, T_encoder_output, head_dim]

        attn = F.scaled_dot_product_attention(  # Shape: [B, num_heads, T, head_dim]
            q,
            k,
            v,
            dropout_p=self.attn_drop,
        )
        x = attn.transpose(1, 2).reshape(B, T, C)  # Shape: [B, T, C]

        x = self.linear_proj(x)
        x = self.dropout(x)
        return x
