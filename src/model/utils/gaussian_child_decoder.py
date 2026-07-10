from __future__ import annotations

import math
from dataclasses import dataclass

import pointops
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from ..types import Gaussians


@dataclass
class GDStyleGaussianChildDecoderCfg:
    enabled: bool = True
    input_dim: int = 512
    hidden_dim: int = 512
    child_feat_dim: int = 320
    num_children: int = 4
    n_frequencies: int = 10
    knn_k: int = 4
    scale_min: float = 1e-6
    scale_init_divisor: float = 1.6
    offset_residual_bound: float = 0.5


def positional_encoding(base_freq: Tensor, x: Tensor) -> Tensor:
    fx = torch.flatten(base_freq[None, None, :, None] * x[:, :, None, :], -2, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


def make_deformable_ring_template(num_children: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a Deformable-DETR-style 2D offset template.

    Returns:
        template: [K, 2], flattened in [head, point] order and truncated to K.
    """
    if num_children < 1:
        raise ValueError(f"num_children must be positive, got {num_children}")
    num_points = max(1, int(math.floor(math.sqrt(num_children))))
    num_heads = int(math.ceil(num_children / num_points))

    theta = torch.arange(num_heads, device=device, dtype=dtype) * (2.0 * math.pi / num_heads)
    directions = torch.stack([theta.cos(), theta.sin()], dim=-1)
    directions = directions / directions.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)

    # Normalize the ring radii here, so the outer ring has radius 1.0.
    # The real metric radius is applied later as d_n * offset_xyz.
    radii = torch.arange(1, num_points + 1, device=device, dtype=dtype) / num_points
    template = directions[:, None, :] * radii[None, :, None]
    return rearrange(template, "h p xy -> (h p) xy")[:num_children]


def quaternion_to_matrix(quaternions: Tensor, eps: float = 1e-8) -> Tensor:
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_child_covariance(scale: Tensor, rotation_xyzw: Tensor) -> Tensor:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


class GDStyleGaussianChildDecoder(nn.Module):
    """Decode child Gaussian offsets from PT features with ReSplat-style updates."""

    def __init__(self, cfg: GDStyleGaussianChildDecoderCfg, sh_degree: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.sh_degree = sh_degree
        self.d_sh = (sh_degree + 1) ** 2
        self.sh_dim = 3 * self.d_sh

        self.in_norm = nn.LayerNorm(cfg.input_dim)
        self.delta_x = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 3 * cfg.num_children),
        )
        self._init_delta_x_as_zero_residual()
        self.register_buffer(
            "offset_template_2d",
            make_deformable_ring_template(cfg.num_children, device=torch.device("cpu"), dtype=torch.float32),
            persistent=False,
        )
        self.skip = nn.Linear(cfg.input_dim, cfg.child_feat_dim)

        pe_dim = 3 * 2 * cfg.n_frequencies if cfg.n_frequencies > 0 else 3
        self.delta_f = nn.Sequential(
            nn.LayerNorm(pe_dim + cfg.input_dim, elementwise_affine=False),
            nn.Linear(pe_dim + cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.child_feat_dim),
        )
        self.out_norm = nn.LayerNorm(cfg.child_feat_dim)

        attr_dim = 3 + 4 + 1 + self.sh_dim
        self.attr_head = nn.Sequential(
            nn.Linear(cfg.child_feat_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, attr_dim),
        )
        nn.init.zeros_(self.attr_head[-1].weight)
        nn.init.zeros_(self.attr_head[-1].bias)

        if cfg.n_frequencies > 0:
            self.register_buffer("frequencies", 2.0 ** torch.arange(cfg.n_frequencies), persistent=False)

    def _init_delta_x_as_zero_residual(self) -> None:
        last = self.delta_x[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("delta_x is expected to end with nn.Linear")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _knn_mean_distance(self, points: Tensor, points_per_view: int | None = None) -> Tensor:
        if points_per_view is not None:
            if points_per_view <= 0:
                raise ValueError(f"points_per_view must be positive, got {points_per_view}")
            if points.shape[1] % points_per_view != 0:
                raise ValueError(
                    "points_per_view must divide the flattened point count: "
                    f"{points_per_view} vs {points.shape[1]}"
                )
            num_views = points.shape[1] // points_per_view
            grouped_points = rearrange(points, "b (v n) c -> (b v) n c", v=num_views, n=points_per_view)
            grouped_scale = self._knn_mean_distance(grouped_points)
            return rearrange(grouped_scale, "(b v) n c -> b (v n) c", b=points.shape[0], v=num_views)

        b, n, _ = points.shape
        if n <= 1:
            return points.new_ones(b, n, 1)

        # Use the same pointops CUDA KNN query as the ReSplat point transformer.
        # It returns only K neighbours per point rather than materializing [N, N]
        # distances, and `points_per_view` above makes each view its own segment.
        num_neighbors = min(self.cfg.knn_k, n - 1)
        with torch.no_grad():
            flat_points = points.detach().float().reshape(b * n, 3).contiguous()
            offsets = torch.arange(1, b + 1, device=points.device, dtype=torch.long) * n
            # Query K + 1 because the first neighbour of each query is itself.
            _, distances = pointops.knn_query(
                num_neighbors + 1,
                flat_points,
                offsets,
                flat_points,
                offsets,
            )
        return (
            distances[:, 1:].mean(dim=-1).reshape(b, n, 1).to(points.dtype).clamp_min(1e-6)
        )

    def _ray_tangent_basis(
        self,
        points: Tensor,
        camera_centers: Tensor,
        camera_rights: Tensor,
        camera_ups: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build a ray-local frame from reference camera axes.

        Shapes:
            points/camera_*: [B, N, 3]
            returns ray_dir/t1/t2: [B, N, 3]
        """
        ray_dir = F.normalize(points.detach() - camera_centers.detach(), dim=-1, eps=1e-6)

        def project_to_tangent(vector: Tensor) -> Tensor:
            return vector - (vector * ray_dir).sum(dim=-1, keepdim=True) * ray_dir

        t1_from_right = project_to_tangent(camera_rights.detach())
        t1_from_up = project_to_tangent(camera_ups.detach())

        x_axis = torch.zeros_like(ray_dir)
        x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(ray_dir)
        y_axis[..., 1] = 1.0
        fallback_axis = torch.where(ray_dir[..., :1].abs() < 0.9, x_axis, y_axis)
        t1_fallback = project_to_tangent(fallback_axis)

        right_valid = t1_from_right.norm(dim=-1, keepdim=True) > 1e-6
        up_valid = t1_from_up.norm(dim=-1, keepdim=True) > 1e-6
        t1 = torch.where(right_valid, t1_from_right, torch.where(up_valid, t1_from_up, t1_fallback))
        t1 = F.normalize(t1, dim=-1, eps=1e-6)
        t2 = F.normalize(torch.cross(ray_dir, t1, dim=-1), dim=-1, eps=1e-6)
        t1 = F.normalize(torch.cross(t2, ray_dir, dim=-1), dim=-1, eps=1e-6)
        return ray_dir, t1, t2

    def _tangent_template_to_world(
        self,
        points: Tensor,
        camera_centers: Tensor | None,
        camera_rights: Tensor | None,
        camera_ups: Tensor | None,
    ) -> Tensor:
        """Map the fixed Deformable-style 2D ring template to tangent-plane xyz.

        Shapes:
            template: [K, 2]
            t1/t2: [B, N, 3]
            bias_xyz: [B, N, K, 3]
        """
        if camera_centers is None or camera_rights is None or camera_ups is None:
            return points.new_zeros(points.shape[0], points.shape[1], self.cfg.num_children, 3)

        if camera_centers.shape != points.shape:
            raise ValueError(f"camera_centers shape mismatch: {tuple(camera_centers.shape)} vs {tuple(points.shape)}")
        if camera_rights.shape != points.shape or camera_ups.shape != points.shape:
            raise ValueError(
                "camera_rights/camera_ups must match points shape: "
                f"{tuple(camera_rights.shape)}, {tuple(camera_ups.shape)}, {tuple(points.shape)}"
            )

        _, t1, t2 = self._ray_tangent_basis(points, camera_centers, camera_rights, camera_ups)
        template = self.offset_template_2d.to(device=points.device, dtype=points.dtype)
        return (
            template[None, None, :, 0:1] * t1[:, :, None, :]
            + template[None, None, :, 1:2] * t2[:, :, None, :]
        )

    def _expand_base_gaussians(self, gaussians: Gaussians, num_children: int) -> dict[str, Tensor]:
        if self.cfg.scale_init_divisor <= 0:
            raise ValueError(f"scale_init_divisor must be positive, got {self.cfg.scale_init_divisor}")
        base_opacity = gaussians.opacities.detach().clamp(1e-6, 1.0 - 1e-6)
        child_opacity = 1.0 - (1.0 - base_opacity).pow(1.0 / num_children)
        base_opacity_raw = torch.logit(child_opacity, eps=1e-6).unsqueeze(-1)
        base_scales = (gaussians.scales.detach() / self.cfg.scale_init_divisor).clamp_min(self.cfg.scale_min)
        base_log_scales = base_scales.log()
        base_sh = rearrange(gaussians.harmonics.detach(), "b n c d -> b n (c d)")
        return {
            "means": repeat(gaussians.means.detach(), "b n c -> b (n k) c", k=num_children),
            "log_scales": repeat(base_log_scales, "b n c -> b (n k) c", k=num_children),
            "rotations": repeat(gaussians.rotations.detach(), "b n c -> b (n k) c", k=num_children),
            "opacities_raw": repeat(base_opacity_raw, "b n c -> b (n k) c", k=num_children),
            "sh": repeat(base_sh, "b n c -> b (n k) c", k=num_children),
        }

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        gaussians: Gaussians,
        camera_centers: Tensor | None = None,
        camera_rights: Tensor | None = None,
        camera_ups: Tensor | None = None,
        points_per_view: int | None = None,
    ) -> dict[str, Tensor | Gaussians]:
        if points.shape[:2] != features.shape[:2]:
            raise ValueError(f"points/features shape mismatch: {tuple(points.shape)} vs {tuple(features.shape)}")
        if gaussians.means.shape[:2] != points.shape[:2]:
            raise ValueError(f"gaussians/points shape mismatch: {tuple(gaussians.means.shape)} vs {tuple(points.shape)}")

        b, n, _ = points.shape
        k = self.cfg.num_children
        in_f = self.in_norm(features)

        knn_scale = self._knn_mean_distance(points, points_per_view=points_per_view)
        bias_xyz = self._tangent_template_to_world(
            points,
            camera_centers,
            camera_rights,
            camera_ups,
        )

        # child_xyz = anchor_xyz + d_n * offset. The template is normalized in
        # make_deformable_ring_template, so the outer ring is one local KNN spacing.
        delta_xyz = knn_scale[:, :, None, :] * torch.tanh(bias_xyz + self.delta_x(in_f).reshape(b, n, k, 3))
        child_means_seed = points.detach()[:, :, None, :] + delta_xyz
        delta_xyz = rearrange(delta_xyz, "b n k xyz -> b (n k) xyz")
        child_means_seed = rearrange(child_means_seed, "b n k xyz -> b (n k) xyz")

        skip_f = repeat(in_f, "b n c -> b (n k) c", k=k)
        if self.cfg.n_frequencies > 0:
            pe = positional_encoding(self.frequencies, child_means_seed)
        else:
            pe = child_means_seed
        delta_f = self.delta_f(torch.cat([pe, skip_f], dim=-1))
        child_features = self.out_norm(self.skip(skip_f) + delta_f)

        delta_attrs = self.attr_head(child_features)
        delta_scales, delta_rotations, delta_opacities, delta_sh = delta_attrs.split(
            (3, 4, 1, self.sh_dim),
            dim=-1,
        )
        base = self._expand_base_gaussians(gaussians, k)

        child_means = child_means_seed
        child_log_scales = base["log_scales"] + delta_scales
        child_scales = child_log_scales.exp().clamp_min(self.cfg.scale_min)
        child_rotations_unnorm = base["rotations"] + delta_rotations
        child_rotations = child_rotations_unnorm / (child_rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
        child_opacities_raw = base["opacities_raw"] + delta_opacities
        child_sh = base["sh"] + delta_sh

        child_gaussians = Gaussians(
            means=child_means,
            covariances=build_child_covariance(child_scales, child_rotations),
            rotations=child_rotations,
            scales=child_scales,
            harmonics=rearrange(child_sh, "b n (c d) -> b n c d", c=3),
            opacities=child_opacities_raw.squeeze(-1).sigmoid(),
        )

        return {
            "gaussians": child_gaussians,
            "features": child_features,
            "delta_means": delta_xyz,
            "knn_scale": knn_scale,
            "delta_attrs": delta_attrs,
        }
