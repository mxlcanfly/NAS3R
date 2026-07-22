from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@dataclass
class HighResolutionGaussianHeadCfg:
    hidden_dim: int
    depth_head_layers: int
    gaussian_head_layers: int


def _make_pixel_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> nn.Sequential:
    if num_layers < 2:
        raise ValueError("Pixel MLP must contain at least two layers.")

    layers = []
    in_channels = input_dim
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        ])
        in_channels = hidden_dim
    layers.append(nn.Conv2d(in_channels, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


class HighResolutionGaussianHead(nn.Module):
    """Predict HR inverse-depth offsets and Gaussian attributes per pixel."""

    def __init__(
        self,
        cfg: HighResolutionGaussianHeadCfg,
        feature_dim: int,
        raw_gaussian_dim: int,
    ) -> None:
        super().__init__()
        self.depth_head = _make_pixel_mlp(
            feature_dim,
            cfg.hidden_dim,
            1,
            cfg.depth_head_layers,
        )
        self.gaussian_head = _make_pixel_mlp(
            feature_dim,
            cfg.hidden_dim,
            raw_gaussian_dim,
            cfg.gaussian_head_layers,
        )

        depth_output = self.depth_head[-1]
        nn.init.zeros_(depth_output.weight)
        nn.init.zeros_(depth_output.bias)

    def forward(
        self,
        enhanced_feat: torch.Tensor,
        lr_depth: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_views, _, height, width = enhanced_feat.shape
        features = rearrange(
            enhanced_feat,
            "b v c h w -> (b v) c h w",
        )
        depth_offset = self.depth_head(features)
        raw_gaussians = self.gaussian_head(features)

        coarse_depth = F.interpolate(
            rearrange(lr_depth, "b v h w -> (b v) () h w"),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).clamp_min(1e-6)
        coarse_disparity = coarse_depth.reciprocal()

        near = rearrange(near, "b v -> (b v) () () ()").clamp_min(1e-6)
        far = rearrange(far, "b v -> (b v) () () ()")
        far = torch.maximum(far, near + 1e-6)
        refined_disparity = torch.clamp(
            coarse_disparity + depth_offset,
            min=far.reciprocal(),
            max=near.reciprocal(),
        )
        hr_depth = refined_disparity.clamp_min(1e-6).reciprocal()

        hr_depth = rearrange(
            hr_depth,
            "(b v) 1 h w -> b v h w",
            b=b,
            v=num_views,
        )
        raw_gaussians = rearrange(
            raw_gaussians,
            "(b v) c h w -> b v c h w",
            b=b,
            v=num_views,
        )
        return hr_depth, raw_gaussians
