import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ...geometry.projection import (
    homogenize_points,
    project_camera_space,
    transform_world2cam,
)


class MultiViewCrossAttention(nn.Module):
    """Geometry-guided local cross-attention between image views."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: int = 3,
        depth_tolerance: float = 0.1,
        query_chunk_size: int = 256,
        residual_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd number.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.depth_tolerance = depth_tolerance
        self.query_chunk_size = query_chunk_size

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.residual_scale = nn.Parameter(
            torch.full((dim,), residual_init)
        )

    def _patch_offsets(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        radius = self.patch_size // 2
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=device,
            dtype=dtype,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack((xx / width, yy / height), dim=-1).reshape(
            -1, 2
        )

    @staticmethod
    def _reference_indices(num_views: int, device: torch.device):
        indices = torch.arange(num_views, device=device)
        return indices[None].expand(num_views, -1)[
            ~torch.eye(num_views, dtype=torch.bool, device=device)
        ].reshape(num_views, num_views - 1)

    def _project_and_sample(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_views, channels, height, width = features.shape
        num_points = points.shape[2]
        num_references = reference_indices.shape[1]

        reference_features = features[:, reference_indices]
        reference_depths = depths[:, reference_indices]
        reference_extrinsics = extrinsics[:, reference_indices]
        reference_intrinsics = intrinsics[:, reference_indices]

        camera_points = transform_world2cam(
            homogenize_points(points)[:, :, None],
            reference_extrinsics[:, :, :, None],
        )[..., :-1]
        projected_depth = camera_points[..., 2]
        projected_xy = project_camera_space(
            camera_points,
            reference_intrinsics[:, :, :, None],
        )

        center_grid = rearrange(
            projected_xy * 2 - 1,
            "b s r n xy -> (b s r) n () xy",
        )
        sampled_depth = F.grid_sample(
            rearrange(reference_depths, "b s r h w -> (b s r) () h w"),
            center_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_depth = rearrange(
            sampled_depth,
            "(b s r) 1 n 1 -> b s r n",
            b=b,
            s=num_views,
            r=num_references,
            n=num_points,
        )

        radius = self.patch_size // 2
        x_margin = radius / width
        y_margin = radius / height
        in_bounds = (
            (projected_xy[..., 0] >= x_margin)
            & (projected_xy[..., 0] <= 1 - x_margin)
            & (projected_xy[..., 1] >= y_margin)
            & (projected_xy[..., 1] <= 1 - y_margin)
        )
        relative_depth_error = (
            (projected_depth - sampled_depth).abs()
            / sampled_depth.abs().clamp_min(1e-6)
        )
        valid = (
            in_bounds
            & projected_depth.isfinite()
            & (projected_depth > 1e-6)
            & sampled_depth.isfinite()
            & (sampled_depth > 1e-6)
            & (relative_depth_error <= self.depth_tolerance)
        )

        patch_xy = projected_xy[..., None, :] + self._patch_offsets(
            height,
            width,
            features.device,
            features.dtype,
        )
        patch_grid = rearrange(
            patch_xy * 2 - 1,
            "b s r n p xy -> (b s r) n p xy",
        )
        sampled_features = F.grid_sample(
            rearrange(
                reference_features,
                "b s r c h w -> (b s r) c h w",
            ),
            patch_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_features = rearrange(
            sampled_features,
            "(b s r) c n p -> b s n r p c",
            b=b,
            s=num_views,
            r=num_references,
            n=num_points,
        )
        valid = rearrange(valid, "b s r n -> b s n r ()").expand(
            -1, -1, -1, -1, self.patch_size**2
        )
        return sampled_features, valid

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        query_has_valid = valid.flatten(-2).any(dim=-1, keepdim=True)
        query = rearrange(
            query,
            "b s n (h d) -> b s n h d",
            h=self.num_heads,
        )
        key = rearrange(
            key,
            "b s n r p (h d) -> b s n h r p d",
            h=self.num_heads,
        )
        value = rearrange(
            value,
            "b s n r p (h d) -> b s n h r p d",
            h=self.num_heads,
        )
        scores = torch.einsum(
            "bsnhd,bsnhrpd->bsnhrp",
            query,
            key,
        ) / math.sqrt(self.head_dim)

        valid = valid[:, :, :, None].expand(
            -1, -1, -1, self.num_heads, -1, -1
        )
        scores = scores.flatten(-2)
        valid = valid.flatten(-2)
        has_valid = valid.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(
            ~valid,
            torch.finfo(scores.dtype).min,
        )
        scores = torch.where(has_valid, scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        attention = attention.unflatten(
            -1, (key.shape[4], key.shape[5])
        )

        update = torch.einsum(
            "bsnhrp,bsnhrpd->bsnhd",
            attention,
            value,
        )
        update = rearrange(update, "b s n h d -> b s n (h d)")
        return self.output_projection(update) * query_has_valid.to(update.dtype)

    def forward(
        self,
        features: torch.Tensor,
        world_points: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        b, num_views, channels, height, width = features.shape
        if num_views < 2:
            return features
        if world_points.shape != (b, num_views, height, width, 3):
            raise ValueError("world_points must match the feature grid.")

        source_features = rearrange(
            features,
            "b v c h w -> b v (h w) c",
        )
        queries = self.query_projection(
            self.query_norm(source_features)
        )
        context_features = self.context_norm(source_features)
        key_features = rearrange(
            self.key_projection(context_features),
            "b v (h w) c -> b v c h w",
            h=height,
            w=width,
        )
        value_features = rearrange(
            self.value_projection(context_features),
            "b v (h w) c -> b v c h w",
            h=height,
            w=width,
        )
        points = rearrange(world_points, "b v h w xyz -> b v (h w) xyz")
        reference_indices = self._reference_indices(num_views, features.device)
        updates = []

        # Projection coordinates are geometric guidance, not optimization targets.
        with torch.no_grad():
            detached_points = points.detach()
            detached_depths = depths.detach()
            detached_extrinsics = extrinsics.detach()
            detached_intrinsics = intrinsics.detach()

        for start in range(0, points.shape[2], self.query_chunk_size):
            end = min(start + self.query_chunk_size, points.shape[2])
            sampled_features, valid = self._project_and_sample(
                detached_points[:, :, start:end],
                torch.cat((key_features, value_features), dim=2),
                detached_depths,
                detached_extrinsics,
                detached_intrinsics,
                reference_indices,
            )
            key, value = sampled_features.split(self.dim, dim=-1)
            updates.append(
                self._attend(
                    queries[:, :, start:end],
                    key,
                    value,
                    valid,
                )
            )

        update = torch.cat(updates, dim=2)
        output = source_features + update * self.residual_scale
        return rearrange(
            output,
            "b v (h w) c -> b v c h w",
            h=height,
            w=width,
        )
