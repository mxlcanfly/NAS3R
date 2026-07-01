from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dpt_block import DPTOutputAdapter, Interpolate


class SpatialFeatureFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, path: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        image_feature = rearrange(image_feature, "b c h w -> b h w c")
        image_feature = self.norm(image_feature)
        image_feature = rearrange(image_feature, "b h w c -> b c h w")
        image_feature = self.project(image_feature)
        if image_feature.shape[-2:] != path.shape[-2:]:
            image_feature = F.interpolate(
                image_feature,
                size=path.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        return self.fuse(torch.cat([path, image_feature], dim=1))


class FusionDPT(DPTOutputAdapter):
    def __init__(
        self,
        *,
        resunet_channels: tuple[int, int, int] = (32, 64, 128),
        **kwargs,
    ) -> None:
        self.resunet_channels = resunet_channels
        super().__init__(head_type="gs_params", **kwargs)

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        feature_dim = self.feature_dim
        c_1_1, c_1_2, c_1_4 = self.resunet_channels
        self.path2_fusion = SpatialFeatureFusion(c_1_4, feature_dim)
        self.path1_fusion = SpatialFeatureFusion(c_1_2, feature_dim)
        self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.direct_img_proj = nn.Sequential(
            nn.Conv2d(c_1_1, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        sr_tokens: List[torch.Tensor],
        resunet_features: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) first."
        h, w = image_size
        n_h = h // (self.stride_level * self.P_H)
        n_w = w // (self.stride_level * self.P_W)

        b, v = sr_tokens[0].shape[:2]
        layers = [sr_tokens[hook] for hook in self.hooks]
        layers = [self.adapt_tokens(layer) for layer in layers]
        layers = [
            rearrange(layer, "b v n c -> (b v) n c")
            for layer in layers
        ]
        layers = [
            rearrange(layer, "bv (nh nw) c -> bv c nh nw", nh=n_h, nw=n_w)
            for layer in layers
        ]
        layers = [self.act_postprocess[idx](layer) for idx, layer in enumerate(layers)]
        layers = [self.scratch.layer_rn[idx](layer) for idx, layer in enumerate(layers)]

        res_1_4 = rearrange(resunet_features["1_4"], "b v c h w -> (b v) c h w")
        res_1_2 = rearrange(resunet_features["1_2"], "b v c h w -> (b v) c h w")
        res_1_1 = rearrange(resunet_features["1_1"], "b v c h w -> (b v) c h w")

        path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_2 = self.path2_fusion(path_2, res_1_4)
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        path_1 = self.path1_fusion(path_1, res_1_2)

        path_1 = self.feat_up(path_1)
        direct_img_feat = self.direct_img_proj(res_1_1)
        if direct_img_feat.shape[-2:] != path_1.shape[-2:]:
            direct_img_feat = F.interpolate(
                direct_img_feat,
                size=path_1.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        path_1 = path_1 + direct_img_feat

        return rearrange(path_1, "(b v) c h w -> b v c h w", b=b, v=v)


def create_fusion_dpt(
    net,
    resunet_channels: tuple[int, int, int] = (32, 64, 128),
    feature_dim: int = 256,
    last_dim: int = 128,
) -> FusionDPT:
    assert net.dec_depth > 9
    dec_depth = net.dec_depth
    dpt = FusionDPT(
        num_channels=feature_dim,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks=[0, dec_depth * 2 // 4, dec_depth * 3 // 4, dec_depth],
        dim_tokens_enc=[net.enc_embed_dim, net.dec_embed_dim, net.dec_embed_dim, net.dec_embed_dim],
        resunet_channels=resunet_channels,
    )
    return dpt
