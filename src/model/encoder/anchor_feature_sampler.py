from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ...geometry.projection import homogenize_points, transform_world2cam, project_camera_space


@dataclass
class AnchorFeatureSamples:
    features: torch.Tensor
    entropy: torch.Tensor | None
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
        entropy_map: torch.Tensor | None = None,
    ) -> AnchorFeatureSamples:
        b, num_views, channels, feat_h, feat_w = feature_map.shape
        num_anchors = anchors.shape[1]

        projected_xy, camera_depth, valid_mask = self._project_anchors(
            anchors,
            extrinsics,
            intrinsics,
        )

        camera_depth_flat = rearrange(camera_depth, "b v n -> b n v")
        valid_mask_flat = rearrange(valid_mask, "b v n -> b n v")

        offsets = self._patch_offsets(feature_map.device, feature_map.dtype)
        pixel_scale = torch.tensor(
            (feat_w, feat_h),
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
        sampled_entropy = None
        if entropy_map is not None:
            if entropy_map.ndim != 4:
                raise ValueError(
                    "Expected entropy_map with shape [B, V, H, W], "
                    f"got {tuple(entropy_map.shape)}."
                )
            if entropy_map.shape[:2] != (b, num_views):
                raise ValueError(
                    "Expected entropy_map batch/view dimensions "
                    f"{(b, num_views)}, got {tuple(entropy_map.shape[:2])}."
                )
            entropy_map = entropy_map.to(
                device=feature_map.device,
                dtype=feature_map.dtype,
            ).unsqueeze(2)
            if entropy_map.shape[-2:] != (feat_h, feat_w):
                entropy_map = F.interpolate(
                    rearrange(entropy_map, "b v c h w -> (b v) c h w"),
                    size=(feat_h, feat_w),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                entropy_map = rearrange(entropy_map, "b v c h w -> (b v) c h w")

            sampled_entropy = F.grid_sample(
                entropy_map,
                patch_grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=True,
            )
            sampled_entropy = rearrange(
                sampled_entropy,
                "(b v) c n p -> b n v (p c)",
                b=b,
                v=num_views,
                n=num_anchors,
            )
            sampled_features = torch.cat([sampled_features, sampled_entropy], dim=-1)

        return AnchorFeatureSamples(
            features=sampled_features,
            entropy=sampled_entropy,
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
