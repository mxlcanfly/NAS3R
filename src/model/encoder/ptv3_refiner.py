from pathlib import Path
import sys

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common.gaussians import build_covariance


def _load_litept(litept_path: str):
    root = Path(litept_path)
    if (root / "litept").exists():
        litept_root = root
    else:
        litept_root = root / "LitePT-main"
    if str(litept_root) not in sys.path:
        sys.path.insert(0, str(litept_root))
    try:
        from litept.model import LitePT
    except Exception as exc:
        raise ImportError(
            f"Failed to import LitePT from {litept_root}. Please check LitePT dependencies."
        ) from exc
    return LitePT


class GaussianLitePTRefiner(nn.Module):
    def __init__(
        self,
        litept_path: str = "/space0/mengxl/LitePT-main",
        feature_channels: int = 32,
        proj_channels: int = 64,
        hidden_channels: int = 128,
        grid_size: float = 0.02,
        sh_degree: int = 0,
    ) -> None:
        super().__init__()
        LitePT = _load_litept(litept_path)
        self.grid_size = grid_size
        self.sh_dim = 3 * ((sh_degree + 1) ** 2)
        self.raw_feature_channels = feature_channels + 3 + 1 + 3 + 3 + 3

        self.proj = nn.Sequential(
            nn.Linear(self.raw_feature_channels, proj_channels),
            nn.LayerNorm(proj_channels),
            nn.GELU(),
        )
        self.litept = LitePT(in_channels=proj_channels)
        self.litept_out_proj = nn.Sequential(
            nn.Linear(72, proj_channels),
            nn.LayerNorm(proj_channels),
            nn.GELU(),
        )

        out_channels = 3 + 3 + 1 + 4 + self.sh_dim
        layers = []
        mlp_in = proj_channels + 3
        for _ in range(4):
            layers.extend([nn.Linear(mlp_in, hidden_channels), nn.GELU()])
            mlp_in = hidden_channels
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.delta_head = nn.Sequential(*layers)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _flatten_inputs(
        self,
        fusion_feature: torch.Tensor,
        render_error: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        ray_direction: torch.Tensor,
        context_image: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                rearrange(fusion_feature, "b v c h w -> (b v h w) c"),
                rearrange(render_error, "b v c h w -> (b v h w) c"),
                rearrange(depth, "b v h w -> (b v h w) 1"),
                rearrange(point_map, "b v h w xyz -> (b v h w) xyz"),
                rearrange(ray_direction, "b v h w xyz -> (b v h w) xyz"),
                rearrange(context_image, "b v c h w -> (b v h w) c"),
            ],
            dim=-1,
        )

    def _aggregate_features(
        self,
        projected_feature: torch.Tensor,
        point_map: torch.Tensor,
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        coord = rearrange(point_map, "b v h w xyz -> (b v h w) xyz").float()
        offset = torch.arange(1, b + 1, device=coord.device, dtype=torch.long) * (v * h * w)
        point = self.litept(
            {
                "coord": coord,
                "grid_size": torch.tensor(self.grid_size, device=coord.device, dtype=coord.dtype),
                "feat": projected_feature.float(),
                "offset": offset,
            }
        )
        litept_feat = point.feat
        if litept_feat.shape[0] != projected_feature.shape[0]:
            if "inverse" not in point:
                raise RuntimeError(
                    "LitePT returned downsampled features without inverse indices."
                )
            litept_feat = litept_feat[point.inverse]
        return projected_feature + self.litept_out_proj(litept_feat.float())

    def forward(
        self,
        fusion_feature: torch.Tensor,
        render_error: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        context_extrinsics: torch.Tensor,
        context_image: torch.Tensor,
        gaussians,
    ):
        b, v, _, h, w = fusion_feature.shape
        camera_center = context_extrinsics[..., :3, 3].unsqueeze(-2).unsqueeze(-2)
        ray_direction = point_map - camera_center
        ray_direction = ray_direction / ray_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        flat_feature = self._flatten_inputs(
            fusion_feature,
            render_error,
            depth,
            point_map,
            ray_direction,
            context_image,
        )
        projected_feature = self.proj(flat_feature.float())
        refined_feature = self._aggregate_features(projected_feature, point_map, b, v, h, w)
        image_feature = rearrange(context_image, "b v c h w -> (b v h w) c").float()
        delta = self.delta_head(torch.cat([refined_feature, image_feature], dim=-1))
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
        delta_rotations = delta[..., cursor:cursor + 4]
        cursor += 4
        delta_sh = delta[..., cursor:cursor + self.sh_dim]

        means = means + delta_means
        scales = (scales + delta_scales).clamp_min(1e-6)
        opacities_raw = torch.logit(opacities.clamp(1e-6, 1 - 1e-6)) + delta_opacities
        opacities = opacities_raw.sigmoid()
        rotations = rotations + delta_rotations
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
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
