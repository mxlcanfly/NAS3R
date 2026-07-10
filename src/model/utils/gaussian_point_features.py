from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from torch import Tensor

from ...geometry.projection import homogenize_points, project, transform_world2cam
from ..types import Gaussians


@dataclass
class ContextGaussianRender:
    color: Tensor
    depth: Tensor
    alpha: Tensor


LR_IMAGE_CHANNELS = 3
LR_RENDER_COLOR_CHANNELS = 3
LR_RENDER_ALPHA_CHANNELS = 1
LR_RENDER_DEPTH_CHANNELS = 1
LR_IMAGE_CHANNEL_START = 0
LR_RENDER_COLOR_CHANNEL_START = LR_IMAGE_CHANNEL_START + LR_IMAGE_CHANNELS
LR_RENDER_ALPHA_CHANNEL = LR_RENDER_COLOR_CHANNEL_START + LR_RENDER_COLOR_CHANNELS
LR_RENDER_DEPTH_CHANNEL = LR_RENDER_ALPHA_CHANNEL + LR_RENDER_ALPHA_CHANNELS
LR_CONTEXT_FEATURE_CHANNELS = LR_RENDER_DEPTH_CHANNEL + LR_RENDER_DEPTH_CHANNELS


@dataclass
class GDCrossAttentionCfg:
    enabled: bool = True
    gaussian_feat_dim: int = 256
    cond_dim: int = 9
    hidden_dim: int = 160
    num_heads: int = 16


class GDGaussianFeatureCrossAttention(nn.Module):
    """Generative-Densification-style per-Gaussian cross attention.

    The LR Gaussian feature is the query. Per-view sampled point features are
    key/value tokens. The returned ``pt_input_feats`` follows GD's convention:
    concat(mlp1(coarse_feature), mlp2(cross_attended_feature)).
    """

    def __init__(self, cfg: GDCrossAttentionCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp1 = nn.Sequential(
            nn.Linear(cfg.gaussian_feat_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            kdim=cfg.cond_dim,
            vdim=cfg.cond_dim,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self._init_linear(self.mlp1)
        self._init_linear(self.mlp2)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, gaussian_feats: Tensor, point_feats: Tensor) -> dict[str, Tensor]:
        if gaussian_feats.ndim != 4:
            raise ValueError(
                "gaussian_feats must have shape [B,V,N,C], got "
                f"{tuple(gaussian_feats.shape)}."
            )
        if point_feats.ndim != 5:
            raise ValueError(
                "point_feats must have shape [B,V,N,V_ctx,C], got "
                f"{tuple(point_feats.shape)}."
            )
        if gaussian_feats.shape[:3] != point_feats.shape[:3]:
            raise ValueError(
                "gaussian_feats and point_feats must share [B,V,N], got "
                f"{tuple(gaussian_feats.shape)} and {tuple(point_feats.shape)}."
            )
        if gaussian_feats.shape[-1] != self.cfg.gaussian_feat_dim:
            raise ValueError(
                f"Expected gaussian feature dim {self.cfg.gaussian_feat_dim}, "
                f"got {gaussian_feats.shape[-1]}."
            )
        if point_feats.shape[-1] != self.cfg.cond_dim:
            raise ValueError(
                f"Expected point condition dim {self.cfg.cond_dim}, got {point_feats.shape[-1]}."
            )

        b, v, n, _, _ = point_feats.shape
        coarse = self.mlp1(gaussian_feats.reshape(b * v * n, gaussian_feats.shape[-1]))
        query = self.norm(coarse).unsqueeze(1)
        key_value = point_feats.reshape(b * v * n, point_feats.shape[-2], point_feats.shape[-1])
        fine = self.cross_att(query, key_value, key_value, need_weights=False)[0].squeeze(1)
        fine = self.mlp2(fine)
        pt_input = torch.cat((coarse, fine), dim=-1)
        return {
            "coarse_feats": coarse.reshape(b, v, n, -1),
            "fine_feats": fine.reshape(b, v, n, -1),
            "pt_input_feats": pt_input.reshape(b, v, n, -1),
        }


def _ensure_render_channel(tensor: Tensor) -> Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected rendered map with shape [B,H,W] or [B,1,H,W], got {tuple(tensor.shape)}.")
    return tensor


def render_gaussians_to_context(
    gaussians: Gaussians,
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Tensor,
    far: Tensor,
    image_shape: tuple[int, int],
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_invariant: bool = True,
) -> ContextGaussianRender:
    """Render a batch of flat Gaussians into each context camera."""
    try:
        from ..decoder.cuda_splatting import render_cuda
    except ModuleNotFoundError as exc:
        if exc.name == "diff_gauss_camera":
            raise RuntimeError(
                "Rendering LR Gaussians for point features requires the "
                "diff_gauss_camera CUDA rasterizer. Install/build the project's "
                "splatting CUDA extension in the active environment first."
            ) from exc
        raise

    b, v, _, _ = extrinsics.shape
    bg = torch.tensor(background_color, dtype=torch.float32, device=extrinsics.device)
    color, depth, alpha = render_cuda(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(intrinsics, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"),
        rearrange(far, "b v -> (b v)"),
        image_shape,
        repeat(bg, "c -> (b v) c", b=b, v=v),
        repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
        repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
        repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
        scale_invariant=scale_invariant,
        return_alpha=True,
    )
    color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
    depth = _ensure_render_channel(depth)
    alpha = _ensure_render_channel(alpha)
    depth = rearrange(depth, "(b v) 1 h w -> b v 1 h w", b=b, v=v)
    alpha = rearrange(alpha, "(b v) 1 h w -> b v 1 h w", b=b, v=v)

    if scale_invariant:
        depth = depth * near[:, :, None, None, None]

    return ContextGaussianRender(color=color, depth=depth, alpha=alpha)


def build_lr_context_feature_stack(
    image_lr: Tensor,
    render: ContextGaussianRender,
) -> Tensor:
    """Stack per-view LR conditioning maps before point sampling."""
    feature_stack = torch.cat(
        [
            image_lr,
            render.color,
            render.alpha,
            render.depth,
        ],
        dim=2,
    )
    if feature_stack.shape[2] != LR_CONTEXT_FEATURE_CHANNELS:
        raise ValueError(
            f"Expected LR context feature stack to have {LR_CONTEXT_FEATURE_CHANNELS} channels, "
            f"got {feature_stack.shape[2]}."
        )
    return feature_stack


def sample_lr_gaussian_point_features(
    points: Tensor,
    feature_stack: Tensor,
    extrinsics: Tensor,
    intrinsics: Tensor,
    padding_mode: str = "border",
) -> Tensor:
    """Sample context-view feature maps at LR Gaussian point projections.

    Args:
        points: World-space LR Gaussian points with shape [B, V_src, N, 3].
        feature_stack: Per-context feature maps [B, V_ctx, C, H, W].
        extrinsics: Context world-to-camera matrices [B, V_ctx, 4, 4].
        intrinsics: Context intrinsics [B, V_ctx, 3, 3].

    Returns:
        Detached point features [B, V_src, N, V_ctx, C + 1], where the final
        channel is absolute z-difference between rendered depth and projected z.
    """
    if points.ndim != 4:
        raise ValueError(f"points must have shape [B, V_src, N, 3], got {tuple(points.shape)}.")
    if feature_stack.ndim != 5:
        raise ValueError(f"feature_stack must have shape [B, V_ctx, C, H, W], got {tuple(feature_stack.shape)}.")

    b, source_views, num_points, _ = points.shape
    _, target_views, channels, _, _ = feature_stack.shape
    if channels <= LR_RENDER_DEPTH_CHANNEL:
        raise ValueError(
            f"feature_stack has {channels} channels, cannot read render depth channel "
            f"{LR_RENDER_DEPTH_CHANNEL}."
        )

    points_bt = points[:, :, None].expand(b, source_views, target_views, num_points, 3)
    points_bt = rearrange(points_bt, "b s t n xyz -> (b s) t n xyz")
    src_w2cs = extrinsics[:, None].expand(b, source_views, target_views, 4, 4)
    src_ixts = intrinsics[:, None].expand(b, source_views, target_views, 3, 3)
    src_w2cs = rearrange(src_w2cs, "b s t i j -> (b s) t i j")
    src_ixts = rearrange(src_ixts, "b s t i j -> (b s) t i j")

    point_xy, _ = project(points_bt, src_w2cs.unsqueeze(2), src_ixts.unsqueeze(2))
    point_xy = rearrange(point_xy, "(b s) t n xy -> b s t n xy", b=b, s=source_views)
    point_grid = rearrange(point_xy * 2 - 1, "b s t n xy -> (b t) (s n) 1 xy")

    sampled = F.grid_sample(
        rearrange(feature_stack, "b t c h w -> (b t) c h w"),
        point_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    sampled = rearrange(
        sampled,
        "(b t) c (s n) 1 -> b s n t c",
        b=b,
        t=target_views,
        s=source_views,
        n=num_points,
    )

    cam_points = transform_world2cam(homogenize_points(points_bt), src_w2cs.unsqueeze(2))
    point_z = rearrange(cam_points[..., 2], "(b s) t n -> b s n t", b=b, s=source_views)
    render_depth = sampled[..., LR_RENDER_DEPTH_CHANNEL]
    z_diff = (render_depth - point_z).abs()

    return torch.cat([sampled, z_diff[..., None]], dim=-1).detach()
