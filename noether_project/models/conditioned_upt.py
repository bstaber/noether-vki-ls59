from __future__ import annotations

import torch
from noether.core.models import Model
from noether.core.schemas.models import UPTConfig
from noether.modeling.models.upt import UPT


class VKIConditionedUPT(Model):
    """Local UPT wrapper that injects per-sample scalar conditioning into the decoder.

    The backbone UPT in noether currently forwards ``condition=None`` to the decoder.
    This wrapper keeps noether package code untouched and adds conditioning locally.
    """

    def __init__(self, model_config: UPTConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = UPT(config=model_config)

    def forward(
        self,
        geometry_batch_idx: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_position: torch.Tensor,
        query_position: torch.Tensor,
        in_scalars: torch.Tensor,
        unbatch_mask_query_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_attn_kwargs, decoder_attn_kwargs = self.backbone.compute_rope_args(
            geometry_batch_idx,
            geometry_position,
            geometry_supernode_idx,
            query_position,
        )

        x = self.backbone.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
        )
        for block in self.backbone.approximator_blocks:
            x, _ = block(x, attn_kwargs=encoder_attn_kwargs)

        queries = self.backbone.pos_embed(query_position)
        # condition = self.scalar_conditioner(in_scalars)

        x = self.backbone.decoder(
            kv=x,
            queries=queries,
            attn_kwargs=decoder_attn_kwargs,
            condition=in_scalars,
        )
        x = self.backbone.norm(x)
        x = self.backbone.prediction_layer(x)

        # If query positions were padded by a collator, flatten and unpad.
        if unbatch_mask_query_position is not None:
            x = x.reshape(-1, x.shape[-1])
            x = x[unbatch_mask_query_position]
        return x
