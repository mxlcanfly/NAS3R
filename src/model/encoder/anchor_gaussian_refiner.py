from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from ..types import Gaussians
from .common.gaussians import build_covariance


@dataclass
class AnchorRefinerOutput:
    gaussians: Gaussians
    initial_depth: Tensor
    refined_depth: Tensor
    source_uv: Tensor


class AnchorGaussianRefiner(nn.Module):
    """Single-pass HR Gaussian refinement from projected multi-view features."""

    def __init__(
        self,
        sh_degree: int,
        num_samples: int = 6,
        lim_dis: float = 0.1,
        feature_channels: int = 128,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if not 0.0 < lim_dis < 1.0:
            raise ValueError(f"lim_dis must be in (0, 1), got {lim_dis}")

        self.num_samples = num_samples
        self.lim_dis = lim_dis
        self.d_sh = (sh_degree + 1) ** 2
        self.attribute_channels = 1 + 3 + 4 + 3 * self.d_sh

        # The fixed seed makes the same low-discrepancy pattern available in
        # training and evaluation. Scrambling avoids Sobol's first boundary point.
        sobol = torch.quasirandom.SobolEngine(
            dimension=2,
            scramble=True,
            seed=0,
        ).draw(num_samples)
        self.register_buffer("sobol_offsets", sobol, persistent=True)

        # 64 Gaussian channels + 64 HAT channels + RGB + coarse disparity.
        self.joint_feature_encoder = nn.Sequential(
            nn.Conv2d(132, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=feature_channels),
            nn.GELU(),
        )

        # Aggregated feature + source uv + initial disparity + local Sobol offset.
        self.anchor_trunk = nn.Sequential(
            nn.Linear(feature_channels + 5, feature_channels),
            nn.LayerNorm(feature_channels),
            nn.GELU(),
            nn.Linear(feature_channels, feature_channels),
            nn.LayerNorm(feature_channels),
            nn.GELU(),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(feature_channels, feature_channels // 2),
            nn.GELU(),
            nn.Linear(feature_channels // 2, 1),
        )
        self.gaussian_head = nn.Sequential(
            nn.Linear(feature_channels, self.attribute_channels * 2),
            nn.GELU(),
            nn.Linear(self.attribute_channels * 2, self.attribute_channels),
        )

        # Refinement starts as an identity mapping.
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.zeros_(self.depth_head[-1].bias)
        nn.init.zeros_(self.gaussian_head[-1].weight)
        nn.init.zeros_(self.gaussian_head[-1].bias)

    def _source_coordinates(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        rows = torch.arange(height, device=device, dtype=dtype)
        cols = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        pixel_xy = torch.stack((grid_x, grid_y), dim=-1)
        coordinates = (
            pixel_xy[:, :, None, :]
            + self.sobol_offsets.to(device=device, dtype=dtype)[None, None]
        )
        normalizer = torch.tensor((width, height), device=device, dtype=dtype)
        coordinates = coordinates / normalizer
        coordinates = rearrange(coordinates, "h w k xy -> (h w k) xy")
        local_offsets = repeat(
            self.sobol_offsets.to(device=device, dtype=dtype),
            "k xy -> (r k) xy",
            r=height * width,
        )
        return coordinates, local_offsets

    @staticmethod
    def _grid_sample_points(feature: Tensor, coordinates: Tensor) -> Tensor:
        """Sample [B,C,H,W] at normalized [0,1] coordinates [B,N,2]."""
        sample_grid = coordinates.mul(2).sub(1).unsqueeze(1)
        sampled = F.grid_sample(
            feature,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.squeeze(2).transpose(1, 2)

    @staticmethod
    def _unproject_to_world(
        coordinates: Tensor,
        depth: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        """Unproject normalized xy and camera z-depth to world coordinates."""
        homogeneous_xy = torch.cat(
            (coordinates, torch.ones_like(coordinates[..., :1])),
            dim=-1,
        )
        camera_rays = torch.einsum(
            "bvij,bvnj->bvni",
            torch.linalg.inv(intrinsics),
            homogeneous_xy,
        )
        camera_points = camera_rays * depth[..., None]
        return (
            torch.einsum(
                "bvij,bvnj->bvni",
                extrinsics[..., :3, :3],
                camera_points,
            )
            + extrinsics[..., None, :3, 3]
        )

    @staticmethod
    def _project_to_views(
        world_points: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        """Project [B,Vsrc,N,3] points into every context view."""
        world_to_camera = torch.linalg.inv(extrinsics)
        camera_points = (
            torch.einsum(
                "boij,bsnj->bsoni",
                world_to_camera[..., :3, :3],
                world_points,
            )
            + world_to_camera[:, None, :, None, :3, 3]
        )
        projected = torch.einsum(
            "boij,bsonj->bsoni",
            intrinsics,
            camera_points,
        )
        z = projected[..., 2:3]
        eps = torch.finfo(projected.dtype).eps
        safe_z = torch.where(
            z.abs() < eps,
            torch.full_like(z, eps),
            z,
        )
        return projected[..., :2] / safe_z

    def _sample_and_average_views(
        self,
        joint_features: Tensor,
        projected_uv: Tensor,
    ) -> Tensor:
        b, source_views, observed_views, num_points = projected_uv.shape[:4]
        channels, height, width = joint_features.shape[2:]
        expanded_features = joint_features[:, None].expand(
            b,
            source_views,
            observed_views,
            channels,
            height,
            width,
        )
        flat_features = rearrange(
            expanded_features,
            "b s o c h w -> (b s o) c h w",
        )
        flat_uv = rearrange(
            projected_uv,
            "b s o n xy -> (b s o) n xy",
        )
        sampled = self._grid_sample_points(flat_features, flat_uv)
        sampled = rearrange(
            sampled,
            "(b s o) n c -> b s o n c",
            b=b,
            s=source_views,
            o=observed_views,
            n=num_points,
        )
        return sampled.mean(dim=2)

    @staticmethod
    def _repeat_parent_attributes(parent: Tensor, num_samples: int) -> Tensor:
        return repeat(parent, "b v r ... -> b v (r k) ...", k=num_samples)

    def forward(
        self,
        gaussian_features: Tensor,
        sr_features: Tensor,
        sr_image: Tensor,
        lr_depth: Tensor,
        lr_gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
    ) -> AnchorRefinerOutput:
        b, views = gaussian_features.shape[:2]
        lr_h, lr_w = lr_depth.shape[-2:]
        hr_h, hr_w = sr_image.shape[-2:]
        if gaussian_features.shape != sr_features.shape:
            raise ValueError(
                "Gaussian and HAT features must have the same shape, got "
                f"{tuple(gaussian_features.shape)} and {tuple(sr_features.shape)}"
            )
        if gaussian_features.shape[2] != 64:
            raise ValueError(
                "Gaussian and HAT features must each have 64 channels, got "
                f"{gaussian_features.shape[2]}"
            )
        if sr_image.shape[:2] != (b, views):
            raise ValueError("SR image batch/view dimensions do not match features")
        if lr_depth.shape[:2] != (b, views):
            raise ValueError(
                f"Expected LR depth {(b, views, lr_h, lr_w)}, "
                f"got {tuple(lr_depth.shape)}"
            )
        if gaussian_features.shape[-2:] != (hr_h, hr_w):
            raise ValueError(
                "Gaussian feature resolution must match the SR image, got "
                f"{tuple(gaussian_features.shape[-2:])} and {(hr_h, hr_w)}"
            )

        coarse_depth_hr = F.interpolate(
            rearrange(lr_depth, "b v h w -> (b v) () h w"),
            size=(hr_h, hr_w),
            mode="bilinear",
            align_corners=False,
        )
        coarse_disparity_hr = coarse_depth_hr.clamp_min(1e-8).reciprocal()
        joint_input = torch.cat(
            (
                rearrange(gaussian_features, "b v c h w -> (b v) c h w"),
                rearrange(sr_features, "b v c h w -> (b v) c h w"),
                rearrange(sr_image, "b v c h w -> (b v) c h w"),
                coarse_disparity_hr,
            ),
            dim=1,
        )
        joint_features = self.joint_feature_encoder(joint_input)
        joint_features = rearrange(
            joint_features,
            "(b v) c h w -> b v c h w",
            b=b,
            v=views,
        )

        source_uv_single, local_offsets_single = self._source_coordinates(
            lr_h,
            lr_w,
            lr_depth.device,
            lr_depth.dtype,
        )
        num_points = source_uv_single.shape[0]
        source_uv = repeat(
            source_uv_single,
            "n xy -> b v n xy",
            b=b,
            v=views,
        )
        local_offsets = repeat(
            local_offsets_single,
            "n xy -> b v n xy",
            b=b,
            v=views,
        )

        initial_depth = self._grid_sample_points(
            rearrange(lr_depth, "b v h w -> (b v) () h w"),
            rearrange(source_uv, "b v n xy -> (b v) n xy"),
        )
        initial_depth = rearrange(
            initial_depth.squeeze(-1),
            "(b v) n -> b v n",
            b=b,
            v=views,
        ).clamp_min(1e-8)
        initial_world_points = self._unproject_to_world(
            source_uv,
            initial_depth,
            extrinsics,
            intrinsics,
        )
        projected_uv = self._project_to_views(
            initial_world_points,
            extrinsics,
            intrinsics,
        )
        anchor_features = self._sample_and_average_views(
            joint_features,
            projected_uv,
        )

        initial_disparity = initial_depth.reciprocal()
        anchor_input = torch.cat(
            (
                anchor_features,
                source_uv,
                initial_disparity[..., None],
                local_offsets,
            ),
            dim=-1,
        )
        anchor_token = self.anchor_trunk(anchor_input)
        raw_depth_shift = self.depth_head(anchor_token).squeeze(-1)

        near = near[..., None].to(initial_depth)
        far = far[..., None].to(initial_depth)
        metric_near_bound = torch.maximum(
            initial_depth * (1.0 - self.lim_dis),
            near,
        )
        metric_near_bound = torch.minimum(metric_near_bound, far)
        metric_far_bound = torch.maximum(
            initial_depth * (1.0 + self.lim_dis),
            near,
        )
        metric_far_bound = torch.minimum(metric_far_bound, far)
        disp_upper = metric_near_bound.clamp_min(1e-8).reciprocal()
        disp_lower = metric_far_bound.clamp_min(1e-8).reciprocal()
        interval = (disp_upper - disp_lower).clamp_min(1e-8)
        base_weight = ((initial_disparity - disp_lower) / interval).clamp(
            1e-6,
            1.0 - 1e-6,
        )
        shift = torch.sigmoid(torch.logit(base_weight) + raw_depth_shift)
        refined_disparity = disp_lower + shift * interval
        refined_depth = refined_disparity.clamp_min(1e-8).reciprocal()
        refined_means = self._unproject_to_world(
            source_uv,
            refined_depth,
            extrinsics,
            intrinsics,
        )

        # The LR adapter returns one surface/sample per pixel in NAS3R-M.
        expected_parent_shape = (b, views, lr_h * lr_w)
        if lr_gaussians.scales.shape[:3] != expected_parent_shape:
            raise ValueError(
                "LR Gaussian batch/view/ray dimensions must match the LR depth, "
                f"expected {expected_parent_shape}, got "
                f"{tuple(lr_gaussians.scales.shape[:3])}"
            )
        if lr_gaussians.scales.shape[3:5] != (1, 1):
            raise ValueError(
                "Anchor refinement currently requires one LR Gaussian per pixel, "
                f"got surface/sample shape {lr_gaussians.scales.shape[3:5]}"
            )
        parent_scales = lr_gaussians.scales[:, :, :, 0, 0].detach()
        parent_rotations = lr_gaussians.rotations[:, :, :, 0, 0].detach()
        parent_opacities = lr_gaussians.opacities[:, :, :, 0, 0].detach()
        parent_harmonics = lr_gaussians.harmonics[:, :, :, 0, 0].detach()
        parent_scales = self._repeat_parent_attributes(
            parent_scales,
            self.num_samples,
        )
        parent_rotations = self._repeat_parent_attributes(
            parent_rotations,
            self.num_samples,
        )
        parent_opacities = self._repeat_parent_attributes(
            parent_opacities,
            self.num_samples,
        )
        parent_harmonics = self._repeat_parent_attributes(
            parent_harmonics,
            self.num_samples,
        )

        parent_log_scales = parent_scales.clamp_min(1e-8).log()
        parent_opacity_logits = torch.logit(
            parent_opacities.clamp(1e-6, 1.0 - 1e-6)
        )
        gaussian_delta = self.gaussian_head(anchor_token)
        delta_opacity, delta_scale, delta_rotation, delta_sh = gaussian_delta.split(
            (1, 3, 4, 3 * self.d_sh),
            dim=-1,
        )

        refined_opacities = (
            parent_opacity_logits + delta_opacity.squeeze(-1)
        ).sigmoid()
        refined_scales = torch.exp(parent_log_scales + delta_scale)
        refined_rotations = F.normalize(
            parent_rotations + delta_rotation,
            dim=-1,
            eps=1e-8,
        )
        refined_harmonics = parent_harmonics + rearrange(
            delta_sh,
            "b v n (rgb sh) -> b v n rgb sh",
            rgb=3,
            sh=self.d_sh,
        )
        refined_covariances = build_covariance(
            refined_scales,
            refined_rotations,
        )

        output_gaussians = Gaussians(
            means=rearrange(refined_means, "b v n xyz -> b (v n) xyz"),
            covariances=rearrange(
                refined_covariances,
                "b v n i j -> b (v n) i j",
            ),
            rotations=rearrange(
                refined_rotations,
                "b v n q -> b (v n) q",
            ),
            scales=rearrange(refined_scales, "b v n xyz -> b (v n) xyz"),
            harmonics=rearrange(
                refined_harmonics,
                "b v n rgb sh -> b (v n) rgb sh",
            ),
            opacities=rearrange(refined_opacities, "b v n -> b (v n)"),
        )
        return AnchorRefinerOutput(
            gaussians=output_gaussians,
            initial_depth=initial_depth,
            refined_depth=refined_depth,
            source_uv=source_uv,
        )
