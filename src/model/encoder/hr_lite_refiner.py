import torch
import torch.nn as nn
from einops import rearrange

from ...geometry.projection import get_world_rays
from .common.gaussian_adapter import GaussianAdapterCfg, Gaussians, UnifiedGaussianAdapter


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MiniUNetRefiner(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
            ResidualConvBlock(feature_channels),
        )
        self.down = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, feature_channels * 2),
            nn.GELU(),
            ResidualConvBlock(feature_channels * 2),
        )
        self.bottleneck = ResidualConvBlock(feature_channels * 2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(feature_channels * 2, feature_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
            ResidualConvBlock(feature_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        x = self.down(skip)
        x = self.bottleneck(x)
        x = self.up(x)
        return self.out(torch.cat([x, skip], dim=1))


class SurfaceBoundedDisparityPredictor(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, lim_dis: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels + 2, hidden_channels, 1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, 1)
        self.act = nn.GELU()
        self.lim_dis = lim_dis

    def forward(
        self,
        feature: torch.Tensor,
        coarse_disparity: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> torch.Tensor:
        near = near.view(-1, 1, 1, 1)
        far = far.view(-1, 1, 1, 1)
        coarse_depth = (1.0 / coarse_disparity.clamp_min(1e-6))
        depth_low = (coarse_depth * (1.0 - self.lim_dis)).clamp(min=near, max=far)
        depth_high = (coarse_depth * (1.0 + self.lim_dis)).clamp(min=near, max=far)
        disp_high = 1.0 / depth_low.clamp_min(1e-6)
        disp_low = 1.0 / depth_high.clamp_min(1e-6)

        feature = self.act(self.conv1(feature))
        feature = torch.cat([feature, disp_low, disp_high], dim=1)
        alpha = torch.sigmoid(self.conv3(self.act(self.conv2(feature))))
        return disp_low + alpha * (disp_high - disp_low)


class HRLiteGaussianGenerator(nn.Module):
    def __init__(
        self,
        litept_path: str = "/space0/mengxl",
        in_channels: int = 32,
        feature_channels: int = 64,
        hidden_channels: int = 128,
        grid_size: float = 0.02,
        sh_degree: int = 4,
        scale_min: float = 0.0,
        scale_max: float = 0.01,
    ) -> None:
        super().__init__()
        del litept_path, grid_size
        self.sh_dim = (sh_degree + 1) ** 2
        self.gaussian_adapter = UnifiedGaussianAdapter(
            GaussianAdapterCfg(
                gaussian_scale_min=scale_min,
                gaussian_scale_max=scale_max,
                sh_degree=sh_degree,
            )
        )

        self.refine_stem = MiniUNetRefiner(in_channels + 3 + 1, feature_channels)
        self.surface_predictor = SurfaceBoundedDisparityPredictor(
            feature_channels,
            feature_channels * 2,
            lim_dis=0.1,
        )
        self.attr_head = nn.Sequential(
            nn.Conv2d(feature_channels + 3, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3 + 4 + 1 + 3 * self.sh_dim, 1),
        )

    def forward(
        self,
        fusion_hr: torch.Tensor,
        depth_hr: torch.Tensor,
        image_sr: torch.Tensor,
        ray_coords_hr: torch.Tensor,
        extrinsics_hr: torch.Tensor,
        intrinsics_hr: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        global_step: int,
        opacity_mapper,
    ) -> Gaussians:
        b, v, _, h, w = fusion_hr.shape
        fusion = rearrange(fusion_hr, "b v c h w -> (b v) c h w")
        image = rearrange(image_sr, "b v c h w -> (b v) c h w").clamp(0, 1)
        disparity = 1.0 / depth_hr.clamp_min(1e-6)
        disparity_in = rearrange(disparity, "b v h w -> (b v) 1 h w")

        refined_feature = self.refine_stem(torch.cat([fusion, image, disparity_in], dim=1))
        refined_disparity = self.surface_predictor(
            refined_feature,
            disparity_in,
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
        ).clamp_min(1e-6)
        refined_depth = 1.0 / refined_disparity

        raw_attrs = self.attr_head(torch.cat([refined_feature, image], dim=1))
        raw_attrs = rearrange(raw_attrs, "(b v) c h w -> b v (h w) c", b=b, v=v)
        raw_scales, raw_rotations, raw_densities, raw_sh = raw_attrs.split(
            (3, 4, 1, 3 * self.sh_dim), dim=-1
        )
        raw_gaussians = torch.cat([raw_scales, raw_rotations, raw_sh], dim=-1)
        depths = rearrange(refined_depth, "(b v) 1 h w -> b v (h w) () 1", b=b, v=v)
        densities = rearrange(raw_densities.sigmoid(), "b v r spp -> b v r () spp")
        opacities = opacity_mapper(densities, global_step)
        ray_origins, ray_directions = get_world_rays(ray_coords_hr, extrinsics_hr, intrinsics_hr)
        means = ray_origins.unsqueeze(3) + ray_directions.unsqueeze(3) * depths[..., None]

        return self.gaussian_adapter.forward(
            means,
            opacities,
            rearrange(raw_gaussians, "b v r c -> b v r () () c"),
        )
