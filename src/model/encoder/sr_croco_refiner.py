from dataclasses import dataclass, field
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from .heads.dpt_head import DPTOutputAdapter_fix


@dataclass
class SRCroCoRefinerCfg:
    enabled: bool = False
    swinir_weight_path: str = ""
    swinir_upscale: int = 4
    swinir_img_size: int = 64
    swinir_window_size: int = 8
    feature_dim: int = 128
    cross_attn_heads: int = 8
    cross_attn_layers: Literal["last", "dpt", "all"] = "dpt"
    dpt_hooks: list[int] = field(default_factory=lambda: [2, 5, 8, 11])


class CrossAttentionTokenRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_mode: Literal["last", "dpt", "all"] = "last",
        dpt_hooks: list[int] | tuple[int, ...] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        self.layer_mode = layer_mode
        self.dpt_hooks = dpt_hooks
        self.sr_from_lr = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_sr = nn.LayerNorm(dim)

    def _selected_indices(self, num_layers: int) -> list[int]:
        if self.layer_mode == "all":
            return list(range(num_layers))
        if self.layer_mode == "dpt":
            return [idx for idx in self.dpt_hooks if 0 <= idx < num_layers]
        return [num_layers - 1]

    def forward(
        self,
        sr_tokens: list[Tensor],
        lr_tokens: list[Tensor],
    ) -> list[Tensor]:
        refined = list(sr_tokens)
        for idx in self._selected_indices(len(sr_tokens)):
            sr = sr_tokens[idx]
            lr = lr_tokens[min(idx, len(lr_tokens) - 1)]
            b, v, sr_len, c = sr.shape

            # Refine each SR context view with its paired LR context view.
            # Cross-view aggregation is left to the CroCo decoder that follows.
            sr_flat = rearrange(sr, "b v l c -> (b v) l c")
            lr_flat = rearrange(lr, "b v l c -> (b v) l c")
            sr_update, _ = self.sr_from_lr(sr_flat, lr_flat, lr_flat, need_weights=False)
            sr_update = self.norm_sr(sr_flat + sr_update)

            refined[idx] = rearrange(sr_update, "(b v) l c -> b v l c", b=b, v=v)
        return refined


class SRFeatureDPT(nn.Module):
    def __init__(
        self,
        dec_embed_dim: int,
        feature_dim: int,
        hooks: list[int] | tuple[int, ...] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        self.hooks = hooks
        self.dpt = DPTOutputAdapter_fix(
            num_channels=feature_dim,
            feature_dim=256,
            last_dim=128,
            hooks=hooks,
            head_type="regression",
        )
        self.dpt.init(dim_tokens_enc=[dec_embed_dim] * len(hooks))

    def forward(self, tokens: list[Tensor], image_size: tuple[int, int]) -> Tensor:
        return self.dpt(tokens, image_size=image_size)
