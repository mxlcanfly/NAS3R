import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common.gaussians import build_covariance
from .pointmlp_aggregator import PointMLPAggregator


class GaussianPointMLPRefiner(nn.Module):
    def __init__(
        self,
        feature_channels: int = 32,
        error_feature_channels: int = 256,
        proj_channels: int = 64,
        hidden_channels: int = 128,
        sh_degree: int = 0,
    ) -> None:
        super().__init__()
        self.sh_dim = 3 * ((sh_degree + 1) ** 2)
        self.gaussian_feature_channels = 3 + 3 + 4 + 1 + self.sh_dim
        self.raw_feature_channels = (
            feature_channels
            + error_feature_channels
            + 3
            + 1
            + 3
            + self.gaussian_feature_channels
        )

        self.proj = nn.Sequential(
            nn.Linear(self.raw_feature_channels, proj_channels),
            nn.LayerNorm(proj_channels),
            nn.GELU(),
        )
        self.pointmlp = PointMLPAggregator(
            channels=proj_channels,
            k_neighbors=16,
            anchor_stride=8,
            num_blocks=2,
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
        feature_error: torch.Tensor,
        render_error: torch.Tensor,
        tr_error_map: torch.Tensor,
        context_image: torch.Tensor,
        gaussian_feature: torch.Tensor,
        num_gaussians_per_pixel: int,
    ) -> torch.Tensor:
        image_feature = torch.cat(
            [
                rearrange(fusion_feature, "b v c h w -> (b v h w) c"),
                rearrange(feature_error, "b v c h w -> (b v h w) c"),
                rearrange(render_error, "b v c h w -> (b v h w) c"),
                rearrange(tr_error_map, "b v c h w -> (b v h w) c"),
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

    def forward(
        self,
        fusion_feature: torch.Tensor,
        feature_error: torch.Tensor,
        render_error: torch.Tensor,
        tr_error_map: torch.Tensor,
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
            feature_error,
            render_error,
            tr_error_map,
            context_image,
            gaussian_feature,
            num_gaussians_per_pixel,
        )
        projected_feature = self.proj(flat_feature.float())
        refined_feature = self.pointmlp(
            projected_feature,
            means,
            b,
            v,
            h,
            w,
            means.shape[3],
            means.shape[4],
        )
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
