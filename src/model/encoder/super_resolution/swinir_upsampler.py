from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

from .network_swinir import SwinIR


class FrozenSwinIRUpsampler(nn.Module):
    def __init__(
        self,
        weight_path: str | Path,
        upscale: int = 4,
        img_size: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.model = SwinIR(
            upscale=upscale,
            in_chans=3,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        checkpoint = self._load_checkpoint(weight_path)
        params = checkpoint["params"] if "params" in checkpoint else checkpoint
        self.model.load_state_dict(params, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _load_checkpoint(weight_path: str | Path):
        try:
            return torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(weight_path, map_location="cpu")

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        self.model.eval()
        *batch, c, h, w = images.shape
        images = rearrange(images, "... c h w -> (...) c h w").clamp(0, 1)

        h_pad = (h // self.window_size + 1) * self.window_size - h
        w_pad = (w // self.window_size + 1) * self.window_size - w
        images = torch.cat([images, torch.flip(images, [2])], 2)[:, :, : h + h_pad, :]
        images = torch.cat([images, torch.flip(images, [3])], 3)[:, :, :, : w + w_pad]

        images_sr = self.model(images)
        images_sr = images_sr[..., : h * self.upscale, : w * self.upscale]
        images_sr = images_sr.clamp(0, 1)
        return images_sr.reshape(*batch, c, h * self.upscale, w * self.upscale)
