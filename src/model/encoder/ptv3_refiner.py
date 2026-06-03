import torch
import torch.nn as nn
from einops import rearrange, repeat

from .common.gaussians import build_covariance


class PointMLPResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.local_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        point_residual = self.point_mlp(x)
        local_feature = rearrange(x, "bv c (h w) -> bv c h w", h=h, w=w)
        local_residual = self.local_mlp(local_feature)
        local_residual = rearrange(local_residual, "bv c h w -> bv c (h w)")
        return self.act(x + point_residual + local_residual)


class PointMLPFeatureRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                PointMLPResidualBlock(out_channels, hidden_channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, feat: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.input_proj(feat)
        for block in self.blocks:
            x = block(x, h, w)
        return x


class GaussianPointMLPRefiner(nn.Module):
    def __init__(
        self,
        feature_channels: int = 32,
        pt_channels: int = 32,
        hidden_channels: int = 128,
        sh_degree: int = 0,
    ) -> None:
        super().__init__()
        self.raw_feature_channels = feature_channels + 3 + 1 + 3 + 3
        self.pt_channels = pt_channels
        self.sh_dim = 3 * ((sh_degree + 1) ** 2)
        self.point_mlp = PointMLPFeatureRefiner(
            self.raw_feature_channels,
            pt_channels,
            hidden_channels=hidden_channels,
            num_blocks=3,
        )

        out_channels = 3 + 3 + 1 + 4 + self.sh_dim

        layers = []
        mlp_in = pt_channels
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
        render_error: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        context_image: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat(
            [
                rearrange(fusion_feature, "b v c h w -> (b v h w) c"),
                rearrange(render_error, "b v c h w -> (b v h w) c"),
                rearrange(depth, "b v h w -> (b v h w) 1"),
                rearrange(point_map, "b v h w xyz -> (b v h w) xyz"),
                rearrange(context_image, "b v c h w -> (b v h w) c"),
            ],
            dim=-1,
        )
        return feat

    def forward(
        self,
        fusion_feature: torch.Tensor,
        render_error: torch.Tensor,
        depth: torch.Tensor,
        point_map: torch.Tensor,
        context_image: torch.Tensor,
        gaussians,
    ):
        b, v, _, h, w = fusion_feature.shape
        feat = self._flatten_inputs(
            fusion_feature, render_error, depth, point_map, context_image
        )
        feat = rearrange(feat.float(), "(b v h w) c -> (b v) c (h w)", b=b, v=v, h=h, w=w)
        refined_feat = self.point_mlp(feat, h, w)
        refined_feat = rearrange(refined_feat, "(b v) c hw -> (b v hw) c", b=b, v=v)
        delta = self.delta_head(refined_feat)
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
