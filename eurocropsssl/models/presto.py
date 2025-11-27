from typing import Literal, cast

import torch
import torch.nn as nn

from eurocropsssl.models.base import Model, ModelBuilder, ModelConfig
from eurocropsssl.models.presto_modules.presto import Presto

PrestoEncoderOut = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Modules to adapt Presto model to pipeline
# https://github.com/nasaharvest/presto/tree/main


class PrestoEncoderWrapper(nn.Module):
    """Class that transforms DataItem into Presto model input.

    Args:
        encoder: Presto Encoder.
        self_supervised: Whether the model is self supervised.
    """

    def __init__(self, encoder: nn.Module, self_supervised: bool):
        super().__init__()
        self.encoder = encoder
        self.self_supervised = self_supervised

    def forward(
        self,
        x: torch.Tensor,
        dates: torch.Tensor,
        dynamic_world: torch.Tensor,
        center: torch.Tensor,
        presto_mask: torch.Tensor,
    ) -> PrestoEncoderOut | tuple[PrestoEncoderOut, torch.Tensor]:
        """Forward pass for Presto Encoder."""
        center = center.to(torch.float32)
        if self.self_supervised:
            enc_out = self.encoder(
                x=x,
                dynamic_world=dynamic_world,
                latlons=center,
                mask=presto_mask,
                month=dates,
                eval_task=False,
            )
            return cast(PrestoEncoderOut, enc_out), dates
        else:
            enc_out = self.encoder(
                x=x,
                dynamic_world=dynamic_world,
                latlons=center,
                mask=presto_mask,
                month=dates,
                eval_task=True,
            )
            return cast(PrestoEncoderOut, enc_out)


class PrestoDecoderWrapper(nn.Module):
    """Class that transforms output from backbone into Presto model decoder input.

    Args:
        decoder: Presto Decoder.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        sup_enc_out: tuple[PrestoEncoderOut, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass that unpacks encoder output and passes to decoder.

        Args:
            sup_enc_out: Tuple containing encoder output (with indices) and month information

        Returns:
            Reconstructed input tensor from decoder (first element of decoder output)
        """
        (x, kept_indices, removed_indices), month = sup_enc_out
        return cast(torch.Tensor, self.decoder(x, kept_indices, removed_indices, month)[0])


class PrestoConfig(ModelConfig):
    """Config for Presto model."""

    channel_encoding: Literal[None] = None
    channels: None = None
    model_builder: Literal["PrestoModelBuilder"] = "PrestoModelBuilder"


class PrestoModelBuilder(ModelBuilder):
    """Presto model builder.

    Args:
        config: Transformer model config.
    """

    def __init__(self, config: PrestoConfig):
        super().__init__(config)

    def build_backbone(self) -> nn.Module:
        """Build Presto backbone for Transformer model."""
        return cast(nn.Module, Presto.construct().encoder)

    def build_self_supervised_head(self) -> nn.Module:
        """Build Presto SSL decoder for Transformer model."""
        return PrestoDecoderWrapper(Presto.construct().decoder)

    def build_classification_head(self, num_classes: int) -> nn.Module:
        """Build linear classification head for Presto Transformer model."""
        return cast(
            nn.Module,
            Presto.construct().construct_finetuning_model(num_outputs=num_classes).head,
        )

    def build_self_supervised_model(self, device: torch.device) -> Model:
        """Build Presto SSL Autoencoder model."""
        backbone = PrestoEncoderWrapper(self.build_backbone(), True)
        head = self.build_self_supervised_head()
        return Model(backbone=backbone, head=head, device=device)

    def build_classification_model(self, num_classes: int, device: torch.device) -> Model:
        """Build Presto classification Transformer model."""
        backbone = PrestoEncoderWrapper(self.build_backbone(), False)

        # Classification Presto is only used in finetuning
        # Thus make sure encoder is trainable but position encodings are not
        backbone.encoder.requires_grad_(True)
        backbone.encoder.pos_embed.requires_grad_(False)
        backbone.encoder.month_embed.requires_grad_(False)

        head = self.build_classification_head(num_classes)
        return Model(backbone=backbone, head=head, device=device)
