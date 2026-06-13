import math

import torch
from torch import nn

from ..types import Gaussians


class AnchorOpacityModulator(nn.Module):
    def __init__(
        self,
        sh_degree: int = 4,
        hidden_dim: int = 32,
        initial_parent_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0 < initial_parent_weight < 1:
            raise ValueError("initial_parent_weight must be in (0, 1).")

        # scale(3) + rotation(4) + opacity(1) + RGB spherical harmonics
        gaussian_feature_dim = 3 + 4 + 1 + 3 * (sh_degree + 1) ** 2
        self.reduce_dim = nn.Sequential(
            nn.Linear(gaussian_feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.generate_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final_layer = self.generate_weight[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
        nn.init.constant_(
            final_layer.bias,
            math.log(initial_parent_weight / (1 - initial_parent_weight)),
        )

    @staticmethod
    def _gaussian_features(gaussians: Gaussians) -> torch.Tensor:
        harmonics = gaussians.harmonics.flatten(start_dim=-2)
        return torch.cat(
            (
                gaussians.scales,
                gaussians.rotations,
                gaussians.opacities[..., None],
                harmonics,
            ),
            dim=-1,
        )

    def forward(
        self,
        lr_gaussians: Gaussians,
        sr_gaussians: Gaussians,
        render_error: torch.Tensor,
    ) -> torch.Tensor:
        lr_features = self._gaussian_features(lr_gaussians)
        num_lr_gaussians = lr_features.shape[1]
        if sr_gaussians.means.shape[1] % num_lr_gaussians != 0:
            raise ValueError(
                "The number of SR Gaussians must be an integer multiple of "
                "the number of LR Gaussians."
            )

        num_children = sr_gaussians.means.shape[1] // num_lr_gaussians
        sr_features = self._gaussian_features(sr_gaussians).reshape(
            sr_gaussians.means.shape[0],
            num_lr_gaussians,
            num_children,
            -1,
        )
        sr_features = sr_features.mean(dim=2)

        if render_error.shape != (*lr_features.shape[:2], 1):
            raise ValueError(
                "render_error must have shape [B, N_lr, 1], got "
                f"{tuple(render_error.shape)}."
            )

        fused_features = self.reduce_dim(
            torch.cat((lr_features, sr_features), dim=-1)
        )
        return self.generate_weight(
            fused_features * render_error.to(dtype=fused_features.dtype)
        ).squeeze(-1).sigmoid()
