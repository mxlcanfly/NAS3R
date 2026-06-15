from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam


@dataclass
class FeatureMultiViewConsistency:
    per_view_consistency_weight: torch.Tensor
    view_projection_valid: torch.Tensor
    occlusion_delta: torch.Tensor
    occlusion_valid_mask: torch.Tensor


class FeatureMultiViewConsistencyEstimator(nn.Module):
    """Measure cosine agreement of projected local feature patches."""

    def __init__(
        self,
        patch_size: int = 4,
        depth_relative_tolerance: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if depth_relative_tolerance <= 0:
            raise ValueError("depth_relative_tolerance must be positive.")
        self.patch_size = patch_size
        self.depth_relative_tolerance = depth_relative_tolerance
        self._stats_print_count = 0

    def _project(
        self,
        anchors: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_views = extrinsics.shape[1]
        anchors_per_view = repeat(anchors, "b n xyz -> b v n xyz", v=num_views)
        camera_points = transform_world2cam(
            homogenize_points(anchors_per_view),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = camera_points[..., -1]
        projected_xy = project_camera_space(
            camera_points,
            intrinsics[:, :, None],
        )
        in_bounds = (
            (projected_xy[..., 0] >= 0)
            & (projected_xy[..., 0] <= 1)
            & (projected_xy[..., 1] >= 0)
            & (projected_xy[..., 1] <= 1)
        )
        return projected_xy, camera_depth, (camera_depth > 1e-6) & in_bounds

    def _sample_feature_patches(
        self,
        features: torch.Tensor,
        projected_xy: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views, _, height, width = features.shape
        radius = (self.patch_size - 1) / 2
        offsets = torch.arange(
            self.patch_size,
            device=features.device,
            dtype=features.dtype,
        ) - radius
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        width_scale = max(width - 1, 1)
        height_scale = max(height - 1, 1)
        offsets = torch.stack(
            (xx / width_scale, yy / height_scale),
            dim=-1,
        ).reshape(-1, 2)
        patch_grid = projected_xy[..., None, :] + offsets
        patch_grid = rearrange(
            patch_grid * 2 - 1,
            "b v n p xy -> (b v) n p xy",
        )
        sampled = F.grid_sample(
            rearrange(features, "b v c h w -> (b v) c h w"),
            patch_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rearrange(
            sampled,
            "(b v) c n p -> b n v (c p)",
            b=b,
            v=num_views,
        )

    @staticmethod
    def _sample_depth(
        depths: torch.Tensor,
        projected_xy: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views = depths.shape[:2]
        grid = rearrange(
            projected_xy * 2 - 1,
            "b v n xy -> (b v) n 1 xy",
        )
        sampled = F.grid_sample(
            rearrange(depths, "b v h w -> (b v) 1 h w"),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rearrange(
            sampled,
            "(b v) 1 n 1 -> b v n",
            b=b,
            v=num_views,
        )

    def forward(
        self,
        anchors: torch.Tensor,
        consistency_features: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        source_view_indices: torch.Tensor,
    ) -> FeatureMultiViewConsistency:
        projected_xy, camera_depth, projection_valid = self._project(
            anchors,
            extrinsics,
            intrinsics,
        )
        sampled_depth = self._sample_depth(depths, projected_xy)
        patch_radius = (self.patch_size - 1) / 2
        x_margin = patch_radius / max(consistency_features.shape[-1] - 1, 1)
        y_margin = patch_radius / max(consistency_features.shape[-2] - 1, 1)
        patch_in_bounds = (
            (projected_xy[..., 0] >= x_margin)
            & (projected_xy[..., 0] <= 1 - x_margin)
            & (projected_xy[..., 1] >= y_margin)
            & (projected_xy[..., 1] <= 1 - y_margin)
        )
        feature_projection_valid = projection_valid & patch_in_bounds
        occlusion_valid_mask = feature_projection_valid & (sampled_depth > 1e-6)
        occlusion_delta = (
            (camera_depth - sampled_depth)
            / (sampled_depth.abs() + 1e-6)
        )
        occlusion_delta = torch.where(
            occlusion_valid_mask,
            occlusion_delta,
            torch.zeros_like(occlusion_delta),
        )
        consistency_patches = self._sample_feature_patches(
            consistency_features,
            projected_xy,
        )
        normalized_patches = F.normalize(
            consistency_patches.float(),
            dim=-1,
            eps=1e-6,
        )
        if source_view_indices.shape != anchors.shape[:2]:
            raise ValueError(
                "Expected source_view_indices with shape "
                f"{tuple(anchors.shape[:2])}, got "
                f"{tuple(source_view_indices.shape)}."
            )
        source_view_indices = source_view_indices.long()
        source_patch_indices = source_view_indices[..., None, None].expand(
            -1,
            -1,
            1,
            normalized_patches.shape[-1],
        )
        source_patches = normalized_patches.gather(
            dim=2,
            index=source_patch_indices,
        ).squeeze(2)
        per_view_similarity = (
            normalized_patches * source_patches[:, :, None]
        ).sum(dim=-1)
        effective_projection_valid = rearrange(
            feature_projection_valid,
            "b v n -> b n v",
        )
        source_view_mask = F.one_hot(
            source_view_indices,
            num_classes=normalized_patches.shape[2],
        ).bool()
        source_projection_valid = effective_projection_valid.gather(
            dim=2,
            index=source_view_indices[..., None],
        )
        cross_view_valid = (
            effective_projection_valid
            & source_projection_valid
            & ~source_view_mask
        )
        cross_view_similarity = per_view_similarity.clamp(0, 1)
        source_consistency = (
            (cross_view_similarity * cross_view_valid).sum(
                dim=2,
                keepdim=True,
            )
            / cross_view_valid.sum(dim=2, keepdim=True).clamp_min(1)
        )
        per_view_consistency_weight = torch.where(
            source_view_mask,
            source_consistency,
            torch.where(
                cross_view_valid,
                cross_view_similarity,
                torch.zeros_like(cross_view_similarity),
            ),
        )
        per_view_consistency_weight = torch.where(
            effective_projection_valid,
            per_view_consistency_weight,
            torch.zeros_like(per_view_similarity),
        )
        return FeatureMultiViewConsistency(
            per_view_consistency_weight=per_view_consistency_weight,
            view_projection_valid=effective_projection_valid,
            occlusion_delta=rearrange(
                occlusion_delta,
                "b v n -> b n v",
            ),
            occlusion_valid_mask=rearrange(
                occlusion_valid_mask,
                "b v n -> b n v",
            ),
        )
