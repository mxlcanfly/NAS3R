from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from .mvca import MultiViewCrossAttention
from .super_resolution.network_swinir import (
    PatchEmbed,
    PatchUnEmbed,
    RSTB,
    Upsample,
)


@dataclass
class SwinFeatureFusionCfg:
    image_feature_dim: int
    embed_dim: int
    output_dim: int
    num_rstb: int
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float
    projection_kernel_size: int
    upsample_scale: int
    upsample_num_feat: int
    use_checkpoint: bool
    mvca_num_heads: int
    mvca_patch_size: int
    mvca_depth_tolerance: float
    mvca_query_chunk_size: int
    mvca_residual_init: float


class SwinFeatureFusion(nn.Module):
    """Fuse an LR image with dense DPT features using a SwinIR RSTB body."""

    def __init__(
        self,
        cfg: SwinFeatureFusionCfg,
        dpt_feature_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if cfg.embed_dim % cfg.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, got "
                f"{cfg.embed_dim} and {cfg.num_heads}."
            )
        if cfg.projection_kernel_size not in (1, 3):
            raise ValueError("projection_kernel_size must be 1 or 3.")
        if cfg.upsample_scale != 4:
            raise ValueError("upsample_scale must be 4 for 64x64 to 256x256.")

        self.cfg = cfg
        self.window_size = cfg.window_size
        projection_padding = cfg.projection_kernel_size // 2

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, cfg.image_feature_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.input_projection = nn.Conv2d(
            dpt_feature_dim + cfg.image_feature_dim,
            cfg.embed_dim,
            cfg.projection_kernel_size,
            1,
            projection_padding,
        )

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=1,
            in_chans=cfg.embed_dim,
            embed_dim=cfg.embed_dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=image_size,
            patch_size=1,
            in_chans=cfg.embed_dim,
            embed_dim=cfg.embed_dim,
            norm_layer=None,
        )
        self.layers = nn.ModuleList([
            RSTB(
                dim=cfg.embed_dim,
                input_resolution=(image_size, image_size),
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=cfg.use_checkpoint,
                img_size=image_size,
                patch_size=1,
                resi_connection="1conv",
            )
            for _ in range(cfg.num_rstb)
        ])
        self.mvca_layers = nn.ModuleList([
            MultiViewCrossAttention(
                dim=cfg.embed_dim,
                num_heads=cfg.mvca_num_heads,
                patch_size=cfg.mvca_patch_size,
                depth_tolerance=cfg.mvca_depth_tolerance,
                query_chunk_size=cfg.mvca_query_chunk_size,
                residual_init=cfg.mvca_residual_init,
            )
            for _ in range(max(cfg.num_rstb - 1, 0))
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.conv_after_body = nn.Conv2d(
            cfg.embed_dim, cfg.embed_dim, 3, 1, 1
        )
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(cfg.embed_dim, cfg.upsample_num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.upsample = Upsample(cfg.upsample_scale, cfg.upsample_num_feat)
        self.output_projection = nn.Conv2d(
            cfg.upsample_num_feat, cfg.output_dim, 3, 1, 1
        )

    def forward(
        self,
        image: torch.Tensor,
        dpt_feature: torch.Tensor,
        world_points: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views, _, height, width = dpt_feature.shape
        image = image.reshape(-1, *image.shape[2:])
        dpt_feature_flat = dpt_feature.reshape(
            b * num_views, *dpt_feature.shape[2:]
        )
        if image.shape[-2:] != (height, width):
            image = F.interpolate(
                image,
                size=dpt_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                "Feature height and width must be divisible by window_size."
            )

        image_feature = self.image_encoder(image)
        cnn_cat_feature = torch.cat(
            (dpt_feature_flat, image_feature), dim=1
        )
        shallow_feature = self.input_projection(cnn_cat_feature)
        feature_size = (height, width)

        tokens = self.patch_embed(shallow_feature)
        mvca_start_layer = len(self.layers) - len(self.mvca_layers)
        for layer_idx, layer in enumerate(self.layers):
            tokens = layer(tokens, feature_size)
            if layer_idx >= mvca_start_layer:
                feature_maps = rearrange(
                    tokens,
                    "(b v) (h w) c -> b v c h w",
                    b=b,
                    v=num_views,
                    h=height,
                    w=width,
                )
                mvca = self.mvca_layers[layer_idx - mvca_start_layer]
                if self.cfg.use_checkpoint and self.training:
                    feature_maps = checkpoint(
                        mvca,
                        feature_maps,
                        world_points,
                        depths,
                        extrinsics,
                        intrinsics,
                        use_reentrant=False,
                    )
                else:
                    feature_maps = mvca(
                        feature_maps,
                        world_points,
                        depths,
                        extrinsics,
                        intrinsics,
                    )
                tokens = rearrange(
                    feature_maps,
                    "b v c h w -> (b v) (h w) c",
                )
        tokens = self.norm(tokens)
        deep_feature = self.patch_unembed(tokens, feature_size)
        fused_feature = self.conv_after_body(deep_feature) + shallow_feature
        output = self.output_projection(
            self.upsample(self.conv_before_upsample(fused_feature))
        )

        output = rearrange(
            output,
            "(b v) c h w -> b v c h w",
            b=b,
            v=num_views,
        )
        return output
