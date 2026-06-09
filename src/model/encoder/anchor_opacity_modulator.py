import math

import torch
from torch import nn


class AnchorOpacityModulator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        initial_parent_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0 < initial_parent_weight < 1:
            raise ValueError("initial_parent_weight must be in (0, 1).")

        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(initial_parent_weight / (1 - initial_parent_weight)),
        )

    def forward(
        self,
        entropy_score: torch.Tensor,
        render_error_score: torch.Tensor,
        parent_opacity: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.stack(
            (entropy_score, render_error_score, parent_opacity),
            dim=-1,
        )
        return self.network(inputs).squeeze(-1).sigmoid()
