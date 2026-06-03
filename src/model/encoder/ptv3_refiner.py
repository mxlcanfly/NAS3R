import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common.gaussians import build_covariance


class GaussianPTV3Refiner(nn.Module):
    def __init__(
        self,
        ptv3_path: str = "/space0/mengxl",
        in_channels: int = 138,
        grid_size: float = 0.02,
        hidden_channels: int = 256,
        refine_rotation: bool = False,
        refine_sh: bool = False,
        sh_degree: int = 0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.sh_dim = 3 * ((sh_degree + 1) ** 2)

        litept_root = Path(ptv3_path)
        if (litept_root / "LitePT-main").exists():
            litept_root = litept_root / "LitePT-main"
        if str(litept_root) not in sys.path:
            sys.path.insert(0, str(litept_root))

        try:
            from litept.model import LitePT
        except Exception as exc:
            raise ImportError(
                "Failed to import LitePT. Please make sure "
                f"{litept_root} is importable and dependencies such as "
                "flash_attn, spconv, torch_scatter, and pointrope are installed."
            ) from exc

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.point_backbone = LitePT(
            in_channels=64,
            enc_patch_size=(128, 128, 128, 128, 128),
            dec_patch_size=(128, 128, 128, 128),
        )
        self.output_proj = nn.Linear(72, 64)

        out_channels = 3 + 3 + 1 + 4 + self.sh_dim

        layers = []
        mlp_in = 64
        for _ in range(3):
            layers.extend([nn.Linear(mlp_in, hidden_channels), nn.GELU()])
            mlp_in = hidden_channels
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.delta_head = nn.Sequential(*layers)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _flatten_inputs(
        self,
        fusion64: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        image_lr: torch.Tensor,
        ray_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, v, c, h, w = fusion64.shape
        coord = rearrange(point_map, "b v h w xyz -> (b v h w) xyz")
        feat = torch.cat(
            [
                rearrange(fusion64, "b v c h w -> (b v h w) c"),
                rearrange(depth, "b v h w -> (b v h w) 1"),
                coord,
                rearrange(image_lr, "b v c h w -> (b v h w) c"),
                rearrange(ray_direction, "b v h w xyz -> (b v h w) xyz"),
            ],
            dim=-1,
        )
        batch = repeat(
            torch.arange(b * v, device=fusion64.device),
            "bv -> (bv hw)",
            hw=h * w,
        )
        return coord, feat, batch

    def forward(
        self,
        fusion64: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        image_lr: torch.Tensor,
        ray_direction: torch.Tensor,
        gaussians,
    ):
        b, v, _, h, w = fusion64.shape
        coord, feat, batch = self._flatten_inputs(
            fusion64, depth, point_map, image_lr, ray_direction
        )
        tmp_feature = self.input_proj(feat.float())
        point = self.point_backbone(
            {
                "coord": coord.float(),
                "feat": tmp_feature,
                "batch": batch,
                "grid_size": self.grid_size,
            }
        )
        out = tmp_feature + self.output_proj(point.feat.float())
        delta = self.delta_head(out)
        delta = rearrange(delta, "(b v h w) c -> b v (h w) c", b=b, v=v, h=h, w=w)

        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

        num_gaussians_per_pixel = means.shape[3] * means.shape[4]
        delta = repeat(delta, "b v hw c -> b v hw r c", r=num_gaussians_per_pixel)
        delta = rearrange(delta, "b v hw (s spp) c -> b v hw s spp c", s=means.shape[3], spp=means.shape[4])

        cursor = 0
        delta_means = delta[..., cursor:cursor + 3]
        cursor += 3
        delta_scales = delta[..., cursor:cursor + 3]
        cursor += 3
        delta_opacities = delta[..., cursor:cursor + 1].squeeze(-1)
        cursor += 1

        means = means + delta_means
        scales = (scales + delta_scales).clamp_min(1e-6)
        opacities_raw = torch.logit(opacities.clamp(1e-6, 1 - 1e-6)) + delta_opacities
        opacities = opacities_raw.sigmoid()


        delta_rotations = delta[..., cursor:cursor + 4]
        cursor += 4
        rotations = rotations + delta_rotations
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

        delta_sh = delta[..., cursor:cursor + self.sh_dim]
        delta_sh = rearrange(delta_sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        harmonics = harmonics + delta_sh

        covariances = build_covariance(scales, rotations)
        return type(gaussians)(
            means=means,
            covariances=covariances,
            rotations=rotations,
            scales=scales,
            harmonics=harmonics,
            opacities=opacities,
        )
