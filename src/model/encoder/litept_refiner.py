import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common.gaussians import build_covariance


class GaussianLitePTRefiner(nn.Module):
    def __init__(
        self,
        litept_path: str = "/space0/mengxl/LitePT-main",
        in_channels: int = 42,
        grid_size: float = 0.02,
        hidden_channels: int = 256,
        refine_rotation: bool = False,
        refine_sh: bool = False,
        sh_degree: int = 0,
    ) -> None:
        super().__init__()
        del refine_rotation, refine_sh
        self.grid_size = grid_size
        self.sh_dim = 3 * ((sh_degree + 1) ** 2)

        litept_root = Path(litept_path)
        if not (litept_root / "litept" / "model.py").is_file():
            litept_root = litept_root / "LitePT-main"
        if str(litept_root) not in sys.path:
            sys.path.insert(0, str(litept_root))

        try:
            from litept.model import LitePT
            from torch_scatter import scatter_mean
        except Exception as exc:
            raise ImportError(
                "Failed to import LitePT. Please make sure "
                f"{litept_root} is importable and flash_attn, spconv, "
                "torch_scatter, and PointROPE are installed."
            ) from exc

        self.scatter_mean = scatter_mean
        self.litept = LitePT(in_channels=in_channels)

        out_channels = 3 + 3 + 1 + 4 + self.sh_dim
        layers = []
        mlp_in = 72
        for _ in range(3):
            layers.extend([nn.Linear(mlp_in, hidden_channels), nn.GELU()])
            mlp_in = hidden_channels
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.delta_head = nn.Sequential(*layers)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _flatten_inputs(
        self,
        fusion_feature: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        image: torch.Tensor,
        ray_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, v, _, h, w = fusion_feature.shape
        coord = rearrange(point_map, "b v h w xyz -> (b v h w) xyz")
        feat = torch.cat(
            [
                rearrange(fusion_feature, "b v c h w -> (b v h w) c"),
                rearrange(depth, "b v h w -> (b v h w) 1"),
                coord,
                rearrange(image, "b v c h w -> (b v h w) c"),
                rearrange(ray_direction, "b v h w xyz -> (b v h w) xyz"),
            ],
            dim=-1,
        )
        batch = repeat(
            torch.arange(b * v, device=fusion_feature.device),
            "bv -> (bv hw)",
            hw=h * w,
        )
        return coord, feat, batch

    def _grid_sample(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        grid_coord = torch.div(
            coord - coord.min(dim=0).values,
            self.grid_size,
            rounding_mode="trunc",
        ).int()
        voxel_key = torch.cat([batch[:, None].int(), grid_coord], dim=-1)
        unique_key, inverse = torch.unique(
            voxel_key,
            sorted=True,
            return_inverse=True,
            dim=0,
        )
        sampled_coord = self.scatter_mean(coord, inverse, dim=0)
        sampled_feat = self.scatter_mean(feat, inverse, dim=0)
        return {
            "coord": sampled_coord.float(),
            "grid_coord": unique_key[:, 1:].int(),
            "feat": sampled_feat.float(),
            "batch": unique_key[:, 0].long(),
            "grid_size": self.grid_size,
        }, inverse

    def forward(
        self,
        fusion_feature: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        image: torch.Tensor,
        ray_direction: torch.Tensor,
        gaussians,
    ):
        b, v, _, h, w = fusion_feature.shape
        coord, feat, batch = self._flatten_inputs(
            fusion_feature, depth, point_map, image, ray_direction
        )
        sampled_point, inverse = self._grid_sample(coord, feat, batch)
        point = self.litept(sampled_point)
        dense_feat = point.feat[inverse]

        delta = self.delta_head(dense_feat.float())
        delta = rearrange(delta, "(b v h w) c -> b v (h w) c", b=b, v=v, h=h, w=w)

        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

        num_gaussians_per_pixel = means.shape[3] * means.shape[4]
        delta = repeat(delta, "b v hw c -> b v hw r c", r=num_gaussians_per_pixel)
        delta = rearrange(
            delta,
            "b v hw (s spp) c -> b v hw s spp c",
            s=means.shape[3],
            spp=means.shape[4],
        )

        cursor = 0
        delta_means = delta[..., cursor:cursor + 3]
        cursor += 3
        delta_scales = delta[..., cursor:cursor + 3]
        cursor += 3
        delta_opacities = delta[..., cursor:cursor + 1].squeeze(-1)
        cursor += 1

        means = means + delta_means
        scales = (scales + delta_scales).clamp_min(1e-6)
        opacities = (
            torch.logit(opacities.clamp(1e-6, 1 - 1e-6)) + delta_opacities
        ).sigmoid()

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
