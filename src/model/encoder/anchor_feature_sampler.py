from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ...geometry.projection import homogenize_points, transform_world2cam, project_camera_space


@dataclass
class AnchorFeatureSamples:
    features: torch.Tensor
    valid_mask: torch.Tensor
    projected_xy: torch.Tensor
    camera_depth: torch.Tensor


class AnchorFeatureSampler(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        self.patch_size = patch_size
        self.padding_mode = padding_mode

    def _patch_offsets(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = (self.patch_size - 1) / 2
        offsets = torch.arange(self.patch_size, device=device, dtype=dtype) - radius
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _project_anchors(
        self,
        anchors: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, num_views = extrinsics.shape[:2]
        num_anchors = anchors.shape[1]

        anchors = repeat(anchors, "b n xyz -> b v n xyz", v=num_views)
        cam_points = transform_world2cam(
            homogenize_points(anchors),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = cam_points[..., -1]
        projected_xy = project_camera_space(cam_points, intrinsics[:, :, None])
        in_front = camera_depth > 1e-6
        in_bounds = (
            (projected_xy[..., 0] >= 0)
            & (projected_xy[..., 0] <= 1)
            & (projected_xy[..., 1] >= 0)
            & (projected_xy[..., 1] <= 1)
        )
        return projected_xy, camera_depth, in_front & in_bounds

    def forward(
        self,
        anchors: torch.Tensor,
        feature_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_map: torch.Tensor | None = None,
        source_view: torch.Tensor | None = None,
        depth_tolerance: float = 0.05,
    ) -> AnchorFeatureSamples:
        b, num_views, channels, feat_h, feat_w = feature_map.shape
        num_anchors = anchors.shape[1]

        projected_xy, camera_depth, valid_mask = self._project_anchors(
            anchors,
            extrinsics,
            intrinsics,
        )

        patch_radius = (self.patch_size - 1) / 2
        x_margin = patch_radius / max(feat_w - 1, 1)
        y_margin = patch_radius / max(feat_h - 1, 1)
        patch_in_bounds = (
            (projected_xy[..., 0] >= x_margin)
            & (projected_xy[..., 0] <= 1 - x_margin)
            & (projected_xy[..., 1] >= y_margin)
            & (projected_xy[..., 1] <= 1 - y_margin)
        )
        valid_mask = valid_mask & patch_in_bounds

        if depth_map is not None:
            if depth_map.shape[:2] != (b, num_views):
                raise ValueError(
                    "depth_map and feature_map must have matching batch/view axes."
                )
            sampled_depth = F.grid_sample(
                rearrange(depth_map, "b v h w -> (b v) 1 h w"),
                rearrange(
                    projected_xy * 2 - 1,
                    "b v n xy -> (b v) n () xy",
                ),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled_depth = rearrange(
                sampled_depth,
                "(b v) 1 n 1 -> b v n",
                b=b,
                v=num_views,
            )
            relative_depth_error = (
                (camera_depth - sampled_depth).abs()
                / sampled_depth.abs().clamp_min(1e-6)
            )
            depth_visible = (
                sampled_depth.isfinite()
                & (sampled_depth > 1e-6)
                & (relative_depth_error <= depth_tolerance)
            )
            valid_mask = valid_mask & depth_visible

        if source_view is not None:
            if source_view.shape != (b, num_anchors):
                raise ValueError(
                    f"Expected source_view shape {(b, num_anchors)}, got "
                    f"{tuple(source_view.shape)}."
                )
            source_mask = F.one_hot(
                source_view,
                num_classes=num_views,
            ).permute(0, 2, 1).bool()
            valid_mask = valid_mask | source_mask

        camera_depth_flat = rearrange(camera_depth, "b v n -> b n v")
        valid_mask_flat = rearrange(valid_mask, "b v n -> b n v")

        offsets = self._patch_offsets(feature_map.device, feature_map.dtype)
        pixel_scale = torch.tensor(
            (max(feat_w - 1, 1), max(feat_h - 1, 1)),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        patch_xy = projected_xy[:, :, :, None] + offsets / pixel_scale
        patch_grid = rearrange(patch_xy * 2 - 1, "b v n p xy -> (b v) n p xy")

        sampled_features = F.grid_sample(
            rearrange(feature_map, "b v c h w -> (b v) c h w"),
            patch_grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        sampled_features = rearrange(
            sampled_features,
            "(b v) c n p -> b n v (p c)",
            b=b,
            v=num_views,
            n=num_anchors,
        )

        return AnchorFeatureSamples(
            features=sampled_features,
            valid_mask=valid_mask_flat,
            projected_xy=rearrange(projected_xy, "b v n xy -> b n v xy"),
            camera_depth=camera_depth_flat,
        )


class AnchorFeatureAggregator(nn.Module):
    def __init__(
        self,
        patch_feature_dim: int = 512,
        num_views: int = 2,
        view_dim: int = 128,
        out_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.patch_feature_dim = patch_feature_dim
        self.num_views = num_views
        self.view_dim = view_dim
        self.view_encoder = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, view_dim),
            nn.LayerNorm(view_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(num_views * view_dim + num_views + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, samples: AnchorFeatureSamples) -> torch.Tensor:
        if samples.features.shape[2] != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} views, got {samples.features.shape[2]}."
            )
        if samples.features.shape[-1] != self.patch_feature_dim:
            raise ValueError(
                f"Expected patch feature dim {self.patch_feature_dim}, "
                f"got {samples.features.shape[-1]}."
            )

        view_features = self.view_encoder(samples.features)
        valid_mask = samples.valid_mask.to(dtype=view_features.dtype)
        view_features = view_features * valid_mask[..., None]
        fused_features = rearrange(view_features, "b n v c -> b n (v c)")
        valid_ratio = valid_mask.mean(dim=-1, keepdim=True)
        fused_features = torch.cat(
            [
                fused_features,
                rearrange(valid_mask, "b n v -> b n v"),
                valid_ratio,
            ],
            dim=-1,
        )
        return self.fusion(fused_features)
