# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from .dpt_block import DPTOutputAdapter, Interpolate


class DPTFeatureAdapter(DPTOutputAdapter):
    """Fuse multi-level single-view encoder tokens into a dense feature map."""

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.feat_up = Interpolate(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        image_size=None,
    ) -> torch.Tensor:
        assert self.dim_tokens_enc is not None, (
            "Need to call init(dim_tokens_enc) function first"
        )
        image_size = self.image_size if image_size is None else image_size
        height, width = image_size
        num_patches_h = height // (self.stride_level * self.P_H)
        num_patches_w = width // (self.stride_level * self.P_W)

        layers = [encoder_tokens[hook] for hook in self.hooks]
        layers = [self.adapt_tokens(layer) for layer in layers]
        layers = [
            rearrange(
                layer,
                "b (nh nw) c -> b c nh nw",
                nh=num_patches_h,
                nw=num_patches_w,
            )
            for layer in layers
        ]
        layers = [
            self.act_postprocess[idx](layer)
            for idx, layer in enumerate(layers)
        ]
        layers = [
            self.scratch.layer_rn[idx](layer)
            for idx, layer in enumerate(layers)
        ]

        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, :layers[2].shape[2], :layers[2].shape[3]
        ]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # DPT path_1 is half-resolution; expose a full-resolution 256-D map.
        return self.feat_up(path_1)


class MultiLevelEncoderFeatureDPT(nn.Module):
    def __init__(self, *, hooks_idx, dim_tokens, feature_dim=256, **kwargs):
        super().__init__()
        self.return_all_layers = True
        self.dpt = DPTFeatureAdapter(
            num_channels=feature_dim,
            feature_dim=feature_dim,
            last_dim=feature_dim // 2,
            hooks=hooks_idx,
            dim_tokens_enc=dim_tokens,
            head_type="gs_params",
            **kwargs,
        )
        self.dpt.init(dim_tokens_enc=dim_tokens)
        del self.dpt.head

    def forward(self, x, img_info):
        return self.dpt(x, image_size=(img_info[0], img_info[1]))


def create_encoder_feature_dpt_head(net):
    feature_dim = 256
    return MultiLevelEncoderFeatureDPT(
        hooks_idx=[0, 1, 2, 3],
        dim_tokens=[net.enc_embed_dim] * 4,
        feature_dim=feature_dim,
    )
