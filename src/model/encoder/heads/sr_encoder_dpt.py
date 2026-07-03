from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from .dpt_block import DPTOutputAdapter, Interpolate


class SRCroCoEncoderDPT(DPTOutputAdapter):
    """Build a dense SR feature map from CroCo encoder tokens only."""

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess
        self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, sr_encoder_tokens: List[torch.Tensor], image_size: tuple[int, int]) -> torch.Tensor:
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) first."
        h, w = image_size
        n_h = h // (self.stride_level * self.P_H)
        n_w = w // (self.stride_level * self.P_W)

        layers = [sr_encoder_tokens[hook] for hook in self.hooks]
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

        path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        path_1 = self.feat_up(path_1)

        b, v = sr_encoder_tokens[0].shape[:2]
        return rearrange(path_1, "(b v) c h w -> b v c h w", b=b, v=v)


def create_sr_encoder_dpt(
    net,
    feature_dim: int = 256,
    last_dim: int = 128,
) -> SRCroCoEncoderDPT:
    enc_depth = net.enc_depth
    dpt = SRCroCoEncoderDPT(
        num_channels=feature_dim,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks=[0, enc_depth * 2 // 4, enc_depth * 3 // 4, enc_depth],
        dim_tokens_enc=[net.enc_embed_dim, net.enc_embed_dim, net.enc_embed_dim, net.enc_embed_dim],
        head_type="gs_params",
    )
    return dpt
