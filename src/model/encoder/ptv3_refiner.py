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
        self.gaussian_feature_channels = 3 + 3 + 4 + 1 + self.sh_dim
        self.raw_feature_channels = (
            feature_channels
            + 1
            + 3
            + self.gaussian_feature_channels
        )

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
        self.delta_head = nn.Sequential(
            nn.Linear(proj_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _flatten_inputs(
        self,
        fusion_feature: torch.Tensor,
        gradient_error: torch.Tensor,
        context_image: torch.Tensor,
        gaussian_feature: torch.Tensor,
        num_gaussians_per_pixel: int,
    ) -> torch.Tensor:
        image_feature = torch.cat(
            [
                rearrange(fusion_feature, "b v c h w -> (b v h w) c"),
                rearrange(gradient_error, "b v c h w -> (b v h w) c"),
                rearrange(context_image, "b v c h w -> (b v h w) c"),
            ],
            dim=-1,
        )
        image_feature = repeat(
            image_feature,
            "n c -> (n r) c",
            r=num_gaussians_per_pixel,
        )
        return torch.cat([image_feature, gaussian_feature], dim=-1)

    def _aggregate_features(
        self,
        projected_feature: torch.Tensor,
        gaussian_means: torch.Tensor,
        b: int,
    ) -> torch.Tensor:
        coord = rearrange(gaussian_means, "b ... xyz -> (b ...) xyz").float()
        points_per_batch = coord.shape[0] // b
        offset = torch.arange(1, b + 1, device=coord.device, dtype=torch.long) * points_per_batch
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
        gradient_error: torch.Tensor,
        context_image: torch.Tensor,
        gaussians,
    ):
        b, v, _, h, w = fusion_feature.shape
        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities
        num_gaussians_per_pixel = means.shape[3] * means.shape[4]

        gaussian_feature = torch.cat(
            [
                means,
                scales,
                rotations,
                opacities.unsqueeze(-1),
                rearrange(harmonics, "... rgb sh -> ... (rgb sh)"),
            ],
            dim=-1,
        )
        gaussian_feature = rearrange(
            gaussian_feature,
            "b v hw s spp c -> (b v hw s spp) c",
        )
        flat_feature = self._flatten_inputs(
            fusion_feature,
            gradient_error,
            context_image,
            gaussian_feature,
            num_gaussians_per_pixel,
        )
        projected_feature = self.proj(flat_feature.float())
        refined_feature = self._aggregate_features(projected_feature, means, b)
        delta = self.delta_head(refined_feature)
        delta = rearrange(
            delta,
            "(b v hw s spp) c -> b v hw s spp c",
            b=b,
            v=v,
            hw=h * w,
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
