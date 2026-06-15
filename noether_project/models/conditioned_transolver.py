from typing import Any

import torch
import torch.nn as nn
from noether.core.models import Model
from noether.core.schemas.models import TransolverConfig
from noether.modeling.models.transolver import Transolver
from noether.modeling.modules.layers import (
    ContinuousSincosEmbed,
    LinearProjection,
)
from noether.modeling.modules.layers.continuous_sincos_embed import (
    ContinuousSincosEmbeddingConfig,
)
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig


class ConditionedTransolverConfig(TransolverConfig):
    """Transolver config extended with aerodynamic data specifications."""

    hidden_dim: int
    position_dim: int
    output_dim: int


class ConditionedTransolver(Model):
    """Transolver conditioned by modulation in the transformer blocks."""

    def __init__(self, model_config: ConditionedTransolverConfig, **kwargs: Any):
        super().__init__(model_config=model_config, **kwargs)

        hidden_dim = model_config.hidden_dim
        position_dim = model_config.position_dim
        output_dim = model_config.output_dim

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(
                hidden_dim=hidden_dim, input_dim=position_dim
            ),
        )

        self.backbone = Transolver(config=model_config)

        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=hidden_dim,
                output_dim=output_dim,
                init_weights="truncnormal002",
                bias=model_config.transformer_block_config.bias,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_kwargs: dict[str, torch.Tensor] | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through conditioned Transolver blocks."""
        x = self.pos_embed(x)
        x = self.backbone(x=x, attn_kwargs=attn_kwargs, condition=condition)
        x = self.out(self.norm(x))
        return x


if __name__ == "__main__":
    """Example of usage of the ConditionedTransolver model."""
    config = ConditionedTransolverConfig(
        name="toto",
        hidden_dim=128,
        depth=12,
        attention_arguments={
            "num_slices": 32,
        },
        transformer_block_config={
            "mlp_expansion_factor": 4,
            "num_heads": 4,
            "condition_dim": 2,
        },
        position_dim=2,
        output_dim=1,
    )
    model = ConditionedTransolver(config)

    num_p = 0
    for p in model.parameters():
        num_p += p.numel()
    print(num_p / 1e6)
