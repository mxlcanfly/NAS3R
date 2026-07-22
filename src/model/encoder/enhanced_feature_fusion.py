from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@dataclass
class EnhancedFeatureFusionCfg:
    gs_feature_dim: int
    encoder_feature_dim: int
    output_dim: int
    num_groups: int


class EnhancedFeatureFusion(nn.Module):
    """Fuse high-resolution encoder and GS-head features per view."""

    def __init__(self, cfg: EnhancedFeatureFusionCfg) -> None:
        super().__init__()
        if cfg.gs_feature_dim % cfg.num_groups != 0:
            raise ValueError("gs_feature_dim must be divisible by num_groups.")
        if cfg.encoder_feature_dim % cfg.num_groups != 0:
            raise ValueError("encoder_feature_dim must be divisible by num_groups.")

        self.gs_adapter = nn.Sequential(
            nn.Conv2d(
                cfg.gs_feature_dim,
                cfg.gs_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(cfg.num_groups, cfg.gs_feature_dim),
            nn.GELU(),
        )
        self.encoder_norm = nn.GroupNorm(
            cfg.num_groups,
            cfg.encoder_feature_dim,
        )
        self.fusion_projection = nn.Conv2d(
            cfg.encoder_feature_dim + cfg.gs_feature_dim,
            cfg.output_dim,
            kernel_size=1,
        )

    def forward(
        self,
        encoder_features: torch.Tensor,
        gs_features: torch.Tensor,
    ) -> torch.Tensor:
        if encoder_features.shape[:2] != gs_features.shape[:2]:
            raise ValueError("Encoder and GS features must have matching views.")

        b, num_views = encoder_features.shape[:2]
        encoder_features = rearrange(
            encoder_features,
            "b v c h w -> (b v) c h w",
        )
        gs_features = rearrange(
            gs_features,
            "b v c h w -> (b v) c h w",
        )
        gs_features = F.interpolate(
            gs_features,
            size=encoder_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gs_features = self.gs_adapter(gs_features)
        encoder_features = self.encoder_norm(encoder_features)
        enhanced_feat = self.fusion_projection(
            torch.cat((encoder_features, gs_features), dim=1)
        )
        return rearrange(
            enhanced_feat,
            "(b v) c h w -> b v c h w",
            b=b,
            v=num_views,
        )
