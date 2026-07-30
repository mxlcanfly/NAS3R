from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

from .hat import HAT

# Resolve the fixed experiment weight relative to the repository, so copying the
# whole NAS3R directory to another machine preserves the integration.
NAS3R_ROOT = Path(__file__).resolve().parents[3]
HAT_WEIGHTS_PATH = (
    NAS3R_ROOT / "pretrained_weights" / "HAT-L_SRx4_ImageNet-pretrain.pth"
)


class FrozenHATLUpsampler(nn.Module):
    """Frozen HAT-L x4 upsampler for [0, 1] RGB context images."""

    upscale = 4

    def __init__(self, chunk_size: int = 4) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not HAT_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"HAT weights not found: {HAT_WEIGHTS_PATH}")
        self.chunk_size = chunk_size

        self.net = HAT(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=(6,) * 12,
            embed_dim=180,
            num_heads=(6,) * 12,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        checkpoint = torch.load(
            HAT_WEIGHTS_PATH,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(checkpoint, dict) or "params_ema" not in checkpoint:
            raise KeyError(
                f"Expected 'params_ema' in HAT checkpoint: {HAT_WEIGHTS_PATH}"
            )
        self.net.load_state_dict(checkpoint["params_ema"], strict=True)
        self.net.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True) -> FrozenHATLUpsampler:
        # The enclosing encoder may enter train mode, but HAT must remain frozen and
        # deterministic. There are no trainable HAT parameters in the optimizer.
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2:] != (3, 64, 64):
            raise ValueError(
                "HAT-L expects [B,V,3,64,64] RGB input, "
                f"got {tuple(images.shape)}"
            )

        batch, views = images.shape[:2]
        flat_images = rearrange(images, "b v c h w -> (b v) c h w").float()
        outputs = []
        feats = []
        # HAT is a fixed image prior. no_grad avoids retaining its large activation
        # graph while still returning a normal tensor usable by later trainable heads.
        with torch.no_grad():
            for chunk in flat_images.split(self.chunk_size):
                result = self.net(chunk)
                # The bundled HAT returns (SR RGB, final upsampling feature).
                sr = result[0] if isinstance(result, tuple) else result
                sr_feat = result[1] if isinstance(result, tuple) else result
                outputs.append(sr.clamp(0, 1))
                feats.append(sr_feat)
        output = torch.cat(outputs, dim=0)
        feats = torch.cat(feats, dim=0)

        expected_shape = (batch * views, 3, 256, 256)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"HAT-L returned {tuple(output.shape)}, expected {expected_shape}"
            )

        output = rearrange(output, "(b v) c h w -> b v c h w", b=batch, v=views)
        feats = rearrange(feats, "(b v) c h w -> b v c h w", b=batch, v=views)
        return output, feats
