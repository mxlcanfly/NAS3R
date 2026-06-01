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

        ptv3_root = Path(ptv3_path)
        if str(ptv3_root) not in sys.path:
            sys.path.insert(0, str(ptv3_root))

        try:
            from PointTransformerV3.model import PointTransformerV3
        except Exception as exc:
            raise ImportError(
                "Failed to import PointTransformerV3. Please make sure "
                f"{ptv3_root}/PointTransformerV3 is importable and dependencies "
                "such as addict, spconv, and torch_scatter are installed."
            ) from exc

        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            enc_channels=(64, 128, 256, 512, 512),
            dec_channels=(64, 128, 256, 512),
            enable_flash=False,
            enc_patch_size=(128, 128, 128, 128, 128),
            dec_patch_size=(128, 128, 128, 128),
        )

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
        point = self.ptv3(
            {
                "coord": coord.float(),
                "feat": feat.float(),
                "batch": batch,
                "grid_size": self.grid_size,
            }
        )
        delta = self.delta_head(point.feat.float())
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
