import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...geometry.projection import project


class LRAnchorSRFeatureSampler(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        self.patch_size = patch_size
        self.padding_mode = padding_mode

    @staticmethod
    def _ensure_channel_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor.unsqueeze(2)
        if tensor.ndim != 5:
            raise ValueError(f"Expected tensor with 4 or 5 dims, got {tuple(tensor.shape)}.")
        return tensor

    def _patch_offsets(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        radius = (self.patch_size - 1) / 2
        offsets = torch.arange(self.patch_size, device=device, dtype=dtype) - radius
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def build_feature_stack(
        self,
        sr_image: torch.Tensor,
        render_color: torch.Tensor,
        render_depth: torch.Tensor,
        render_alpha: torch.Tensor,
        render_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        render_depth = self._ensure_channel_dim(render_depth)
        render_alpha = self._ensure_channel_dim(render_alpha)
        if render_error is None:
            render_error = sr_image - render_color
        return torch.cat(
            [
                sr_image,
                render_color,
                render_alpha,
                render_depth,
                render_error,
            ],
            dim=2,
        )

    def forward(
        self,
        anchors: torch.Tensor,
        feature_stack: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.ndim != 4:
            raise ValueError(f"anchors must have shape [B, V, N, 3], got {tuple(anchors.shape)}.")
        if feature_stack.ndim != 5:
            raise ValueError(
                "feature_stack must have shape [B, V, C, H, W], got "
                f"{tuple(feature_stack.shape)}."
            )

        b, target_views, channels, feat_h, feat_w = feature_stack.shape
        source_views, num_anchors = anchors.shape[1:3]
        if extrinsics.shape[:2] != (b, target_views) or intrinsics.shape[:2] != (b, target_views):
            raise ValueError(
                "extrinsics/intrinsics must match feature_stack [B, V], got "
                f"{tuple(extrinsics.shape)}, {tuple(intrinsics.shape)}, and {tuple(feature_stack.shape)}."
            )

        points = anchors[:, :, None].expand(b, source_views, target_views, num_anchors, 3)
        points = rearrange(points, "b s t n xyz -> (b s) t n xyz")
        src_w2cs = extrinsics[:, None].expand(b, source_views, target_views, 4, 4)
        src_ixts = intrinsics[:, None].expand(b, source_views, target_views, 3, 3)
        src_w2cs = rearrange(src_w2cs, "b s t i j -> (b s) t i j")
        src_ixts = rearrange(src_ixts, "b s t i j -> (b s) t i j")

        point_xy, _ = project(points, src_w2cs.unsqueeze(2), src_ixts.unsqueeze(2))
        point_xy = rearrange(
            point_xy,
            "(b s) t n xy -> b s t n xy",
            b=b,
            s=source_views,
        )

        offsets = self._patch_offsets(feature_stack.device, feature_stack.dtype)
        pixel_scale = point_xy.new_tensor((max(feat_w - 1, 1), max(feat_h - 1, 1)))
        patch_xy = point_xy[:, :, :, :, None] + offsets / pixel_scale
        patch_grid = rearrange(
            patch_xy * 2 - 1,
            "b s t n p xy -> (b t) (s n) p xy",
        )
        sampled = F.grid_sample(
            rearrange(feature_stack, "b t c h w -> (b t) c h w"),
            patch_grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        sampled = rearrange(
            sampled,
            "(b t) c (s n) p -> b s n t (p c)",
            b=b,
            t=target_views,
            s=source_views,
            n=num_anchors,
        )
        return sampled.detach()
