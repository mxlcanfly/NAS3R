from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam


@dataclass
class MultiViewConsistency:
    probability: torch.Tensor
    has_valid_pair: torch.Tensor
    view_visibility: torch.Tensor


@dataclass
class FeatureMultiViewConsistency:
    similarity: torch.Tensor
    consistency_weight: torch.Tensor
    has_valid_pair: torch.Tensor
    view_visibility: torch.Tensor
    sampled_features: torch.Tensor
    aggregated_features: torch.Tensor


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

    def _print_similarity_stats(
        self,
        similarity: torch.Tensor,
        consistency_weight: torch.Tensor,
        has_valid_pair: torch.Tensor,
    ) -> None:
        return

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
        aggregation_features: torch.Tensor,
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
        relative_depth_error = (
            (sampled_depth - camera_depth).abs()
            / torch.maximum(
                sampled_depth.abs(),
                camera_depth.abs(),
            ).clamp_min(1e-6)
        )
        view_visibility = (
            projection_valid
            & patch_in_bounds
            & (sampled_depth > 1e-6)
            & (relative_depth_error <= self.depth_relative_tolerance)
        )
        consistency_patches = self._sample_feature_patches(
            consistency_features,
            projected_xy,
        )
        aggregation_patches = self._sample_feature_patches(
            aggregation_features,
            projected_xy,
        )
        normalized_patches = F.normalize(
            consistency_patches.float(),
            dim=-1,
            eps=1e-6,
        )
        similarity_sum = torch.zeros(
            anchors.shape[:2],
            device=anchors.device,
            dtype=torch.float32,
        )
        valid_pair_count = torch.zeros_like(similarity_sum)

        for first_view in range(consistency_features.shape[1]):
            for second_view in range(
                first_view + 1,
                consistency_features.shape[1],
            ):
                pair_valid = (
                    view_visibility[:, first_view]
                    & view_visibility[:, second_view]
                )
                pair_similarity = (
                    normalized_patches[:, :, first_view]
                    * normalized_patches[:, :, second_view]
                ).sum(dim=-1)
                similarity_sum += pair_similarity * pair_valid
                valid_pair_count += pair_valid

        has_valid_pair = valid_pair_count > 0
        similarity = similarity_sum / valid_pair_count.clamp_min(1)
        similarity = torch.where(
            has_valid_pair,
            similarity,
            torch.full_like(similarity, -1.0),
        )

        if source_view_indices.shape != anchors.shape[:2]:
            raise ValueError(
                "Expected source_view_indices with shape "
                f"{tuple(anchors.shape[:2])}, got "
                f"{tuple(source_view_indices.shape)}."
            )
        source_view_indices = source_view_indices.long()
        source_mask = F.one_hot(
            source_view_indices,
            num_classes=consistency_features.shape[1],
        ).bool()
        visible_by_anchor = rearrange(
            view_visibility,
            "b v n -> b n v",
        )

        # The source view is the direct observation from which the anchor was
        # unprojected, so it always remains in the fusion with unit weight.
        effective_visibility = visible_by_anchor | source_mask

        # Other views are weighted by positive cosine agreement. Negative
        # agreement suppresses them instead of subtracting their features.
        consistency_weight = similarity.clamp(0, 1)
        consistency_weight = torch.where(
            has_valid_pair,
            consistency_weight,
            torch.zeros_like(consistency_weight),
        )[..., None]
        self._print_similarity_stats(
            similarity,
            consistency_weight[..., 0],
            has_valid_pair,
        )
        aggregation_weights = torch.where(
            source_mask,
            torch.ones_like(
                visible_by_anchor,
                dtype=aggregation_patches.dtype,
            ),
            consistency_weight.to(aggregation_patches.dtype),
        )
        aggregation_weights = aggregation_weights * effective_visibility.to(
            aggregation_patches.dtype
        )
        aggregated_features = (
            aggregation_patches * aggregation_weights[..., None]
        ).sum(dim=2) / aggregation_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        return FeatureMultiViewConsistency(
            similarity=similarity[..., None],
            consistency_weight=consistency_weight[..., 0, None],
            has_valid_pair=has_valid_pair[..., None],
            view_visibility=effective_visibility,
            sampled_features=aggregation_patches,
            aggregated_features=aggregated_features,
        )


class MultiViewConsistencyEstimator(nn.Module):
    """Estimate per-anchor confidence from multi-view RGB and depth agreement."""

    def __init__(
        self,
        patch_size: int = 3,
        color_temperature: float = 0.15,
        depth_relative_tolerance: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer.")
        if color_temperature <= 0:
            raise ValueError("color_temperature must be positive.")
        if depth_relative_tolerance <= 0:
            raise ValueError("depth_relative_tolerance must be positive.")

        self.patch_size = patch_size
        self.color_temperature = color_temperature
        self.depth_relative_tolerance = depth_relative_tolerance

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
        projected_xy = project_camera_space(camera_points, intrinsics[:, :, None])
        in_bounds = (
            (projected_xy[..., 0] >= 0)
            & (projected_xy[..., 0] <= 1)
            & (projected_xy[..., 1] >= 0)
            & (projected_xy[..., 1] <= 1)
        )
        return projected_xy, camera_depth, (camera_depth > 1e-6) & in_bounds

    def _sample_rgb_patches(
        self,
        images: torch.Tensor,
        projected_xy: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views, _, height, width = images.shape
        radius = self.patch_size // 2
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=images.device,
            dtype=images.dtype,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        offsets = torch.stack((xx / width, yy / height), dim=-1).reshape(-1, 2)
        patch_grid = projected_xy[..., None, :] + offsets
        patch_grid = rearrange(patch_grid * 2 - 1, "b v n p xy -> (b v) n p xy")
        sampled = F.grid_sample(
            rearrange(images, "b v c h w -> (b v) c h w"),
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

    def _sample_depth(
        self,
        depths: torch.Tensor,
        projected_xy: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views = depths.shape[:2]
        grid = rearrange(projected_xy * 2 - 1, "b v n xy -> (b v) n 1 xy")
        sampled = F.grid_sample(
            rearrange(depths, "b v h w -> (b v) 1 h w"),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rearrange(sampled, "(b v) 1 n 1 -> b v n", b=b, v=num_views)

    @torch.no_grad()
    def forward(
        self,
        anchors: torch.Tensor,
        images: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> MultiViewConsistency:
        images = images.detach().float()
        if images.amin().item() < 0:
            images = images * 0.5 + 0.5
        images = images.clamp(0, 1)

        projected_xy, camera_depth, projection_valid = self._project(
            anchors.detach().float(),
            extrinsics.detach().float(),
            intrinsics.detach().float(),
        )
        sampled_depth = self._sample_depth(depths.detach().float(), projected_xy)
        relative_depth_error = (
            (sampled_depth - camera_depth).abs()
            / torch.maximum(sampled_depth.abs(), camera_depth.abs()).clamp_min(1e-6)
        )
        view_visibility = (
            projection_valid
            & (sampled_depth > 1e-6)
            & (relative_depth_error <= self.depth_relative_tolerance)
        )

        rgb_patches = self._sample_rgb_patches(images, projected_xy)
        probability_sum = torch.zeros(
            anchors.shape[0],
            anchors.shape[1],
            device=anchors.device,
            dtype=torch.float32,
        )
        valid_pair_count = torch.zeros_like(probability_sum)
        num_views = images.shape[1]
        for first_view in range(num_views):
            for second_view in range(first_view + 1, num_views):
                pair_valid = (
                    view_visibility[:, first_view]
                    & view_visibility[:, second_view]
                )
                color_error = (
                    rgb_patches[:, :, first_view]
                    - rgb_patches[:, :, second_view]
                ).abs().mean(dim=-1)
                pair_probability = torch.exp(-color_error / self.color_temperature)
                probability_sum += pair_probability * pair_valid
                valid_pair_count += pair_valid

        has_valid_pair = valid_pair_count > 0
        probability = probability_sum / valid_pair_count.clamp_min(1)
        probability = probability * has_valid_pair
        return MultiViewConsistency(
            probability=probability[..., None],
            has_valid_pair=has_valid_pair[..., None],
            view_visibility=rearrange(view_visibility, "b v n -> b n v"),
        )
