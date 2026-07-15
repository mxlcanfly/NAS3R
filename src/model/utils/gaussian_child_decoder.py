from __future__ import annotations

import math
from dataclasses import dataclass

import torch
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
    scale_min: float = 1e-6
    scale_max: float = 0.3
    scale_init_divisor: float = 1.6
    offset_residual_bound: float = 0.5
    num_rings: int | None = None
    max_radius_pixel: float = 0.95
    hammersley_eps: float = 0.05


def positional_encoding(base_freq: Tensor, x: Tensor) -> Tensor:
    fx = torch.flatten(base_freq[None, None, :, None] * x[:, :, None, :], -2, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


def build_quarter_ring_uv_bias(
    num_children: int,
    num_rings: int | None = None,
    max_radius_pixel: float = 0.95,
    eps: float = 1e-4,
    return_pre_sigmoid: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build lower-right quarter-ring offsets in LR-pixel coordinates."""
    if num_children < 1:
        raise ValueError(f"num_children must be positive, got {num_children}")
    if num_rings is None:
        num_rings = max(1, math.ceil(math.sqrt(num_children)))
    if num_rings < 1 or num_rings > num_children:
        raise ValueError(f"num_rings must be in [1, num_children], got {num_rings}")
    if not 0.0 < max_radius_pixel < 1.0:
        raise ValueError(f"max_radius_pixel must be in (0, 1), got {max_radius_pixel}")
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")

    base, remainder = divmod(num_children, num_rings)
    rings = []
    for ring_idx in range(num_rings):
        count = base + int(ring_idx < remainder)
        radius = max_radius_pixel * (ring_idx + 1) / num_rings
        theta = (torch.arange(count, dtype=dtype) + 0.5) / count * (math.pi / 2.0)
        rings.append(radius * torch.stack((theta.cos(), theta.sin()), dim=-1))

    template_uv = torch.cat(rings, dim=0).clamp(eps, 1.0 - eps)
    return torch.logit(template_uv) if return_pre_sigmoid else template_uv


def build_hammersley_uv_bias(
    num_children: int,
    eps: float = 0.05,
    return_pre_sigmoid: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a deterministic Hammersley template inside one LR pixel cell."""
    if num_children < 1:
        raise ValueError(f"num_children must be positive, got {num_children}")
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")

    def radical_inverse_base2(index: int) -> float:
        value = 0.0
        inv_base = 0.5
        while index:
            value += (index & 1) * inv_base
            index >>= 1
            inv_base *= 0.5
        return value

    indices = torch.arange(num_children, dtype=dtype)
    u = (indices + 0.5) / num_children
    v = torch.tensor(
        [radical_inverse_base2(index) for index in range(num_children)],
        dtype=dtype,
    )
    unit_uv = torch.stack((u, v), dim=-1)
    template_uv = eps + (1.0 - 2.0 * eps) * unit_uv
    return torch.logit(template_uv) if return_pre_sigmoid else template_uv


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
        self._init_delta_x_hammersley_bias()
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

    def _init_delta_x_hammersley_bias(self) -> None:
        last = self.delta_x[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("delta_x is expected to end with nn.Linear")
        nn.init.zeros_(last.weight)
        uv_bias = build_hammersley_uv_bias(
            num_children=self.cfg.num_children,
            eps=self.cfg.hammersley_eps,
            return_pre_sigmoid=True,
            dtype=last.bias.dtype,
        )
        uvz_bias = torch.zeros(
            self.cfg.num_children,
            3,
            device=last.bias.device,
            dtype=last.bias.dtype,
        )
        uvz_bias[:, :2] = uv_bias
        with torch.no_grad():
            last.bias.copy_(uvz_bias.reshape(-1))

    def initial_local_offset_uv(self) -> Tensor:
        """Return the Hammersley initialization template in LR-pixel units."""
        last = self.delta_x[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("delta_x is expected to end with nn.Linear")
        return build_hammersley_uv_bias(
            num_children=self.cfg.num_children,
            eps=self.cfg.hammersley_eps,
            return_pre_sigmoid=False,
            dtype=last.bias.dtype,
        ).to(last.bias.device)

    @staticmethod
    def _unproject_child_uv(
        child_uv: Tensor,
        child_depths: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        points_per_view: int,
    ) -> Tensor:
        """Unproject child UV and child z-depth into world coordinates."""
        b, total_points, k, _ = child_uv.shape
        if total_points % points_per_view != 0:
            raise ValueError("points_per_view must divide the flattened point count")
        num_views = total_points // points_per_view
        child_uv = rearrange(child_uv, "b (v n) k xy -> b v n k xy", v=num_views, n=points_per_view)
        child_depths = rearrange(
            child_depths,
            "b (v n) k one -> b v n k one",
            v=num_views,
            n=points_per_view,
        )
        pixel_h = torch.cat((child_uv, torch.ones_like(child_uv[..., :1])), dim=-1)
        rays = torch.einsum("bvij,bvnkj->bvnki", torch.linalg.inv(intrinsics), pixel_h)
        camera_points = rays * child_depths
        camera_points_h = torch.cat((camera_points, torch.ones_like(camera_points[..., :1])), dim=-1)
        world_points = torch.einsum("bvij,bvnkj->bvnki", extrinsics, camera_points_h)[..., :3]
        return rearrange(world_points, "b v n k xyz -> b (v n) k xyz")

    def _expand_base_gaussians(self, gaussians: Gaussians, num_children: int) -> dict[str, Tensor]:
        if self.cfg.scale_init_divisor <= 0:
            raise ValueError(f"scale_init_divisor must be positive, got {self.cfg.scale_init_divisor}")
        if self.cfg.scale_max <= self.cfg.scale_min:
            raise ValueError(
                "scale_max must be greater than scale_min, got "
                f"{self.cfg.scale_max} <= {self.cfg.scale_min}"
            )
        base_opacity = gaussians.opacities.detach().clamp(1e-6, 1.0 - 1e-6)
        base_opacity_raw = torch.logit(base_opacity, eps=1e-6).unsqueeze(-1)
        base_scales = (gaussians.scales.detach() / self.cfg.scale_init_divisor).clamp_min(self.cfg.scale_min)
        base_sh = rearrange(gaussians.harmonics.detach(), "b n c d -> b n (c d)")
        return {
            "means": repeat(gaussians.means.detach(), "b n c -> b (n k) c", k=num_children),
            "scales": repeat(base_scales, "b n c -> b (n k) c", k=num_children),
            "rotations": repeat(gaussians.rotations.detach(), "b n c -> b (n k) c", k=num_children),
            "opacities_raw": repeat(base_opacity_raw, "b n c -> b (n k) c", k=num_children),
            "sh": repeat(base_sh, "b n c -> b (n k) c", k=num_children),
        }

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        gaussians: Gaussians,
        parent_uv: Tensor,
        parent_depths: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
        points_per_view: int,
    ) -> dict[str, Tensor | Gaussians]:
        if points.shape[:2] != features.shape[:2]:
            raise ValueError(f"points/features shape mismatch: {tuple(points.shape)} vs {tuple(features.shape)}")
        if gaussians.means.shape[:2] != points.shape[:2]:
            raise ValueError(f"gaussians/points shape mismatch: {tuple(gaussians.means.shape)} vs {tuple(points.shape)}")

        b, n, _ = points.shape
        k = self.cfg.num_children
        in_f = self.in_norm(features)

        predicted_offset_uvz = self.delta_x(in_f).reshape(b, n, k, 3)
        local_offset_uv = torch.sigmoid(predicted_offset_uvz[..., :2])
        predicted_delta_z = predicted_offset_uvz[..., 2:3]
        h, w = image_shape
        pixel_size = points.new_tensor((1.0 / w, 1.0 / h))
        child_uv = parent_uv.detach()[:, :, None, :] + local_offset_uv * pixel_size
        child_depths = parent_depths.detach()[:, :, None, None] + predicted_delta_z
        child_means_seed = self._unproject_child_uv(
            child_uv,
            child_depths,
            extrinsics,
            intrinsics,
            points_per_view,
        )
        delta_xyz = child_means_seed - points.detach()[:, :, None, :]
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
        child_scales = (base["scales"].detach() + delta_scales).clamp(
            min=self.cfg.scale_min,
            max=self.cfg.scale_max,
        )
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
            "local_offset_uv": local_offset_uv,
            "predicted_delta_z": predicted_delta_z,
            "child_depths": child_depths,
            "delta_attrs": delta_attrs,
        }
