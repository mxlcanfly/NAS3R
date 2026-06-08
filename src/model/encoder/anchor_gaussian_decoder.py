import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from ..types import Gaussians
from .common.gaussians import build_covariance


class AnchorGaussianResidualDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        gaussians_per_anchor: int = 8,
        sh_degree: int = 4,
        offset_radius: float = 0.75,
        ring_radius_beta: float = 0.2,
        scale_residual_range: float = 0.5,
        opacity_residual_range: float = 2.0,
        rotation_residual_range: float = 0.1,
        sh_residual_range: float = 0.1,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.gaussians_per_anchor = gaussians_per_anchor
        self.sh_dim = (sh_degree + 1) ** 2
        self.offset_radius = offset_radius
        self.ring_radius_beta = ring_radius_beta
        # Kept for config compatibility. The decoder now follows the ReSplat-style
        # direct residual update for scale, opacity, rotation, and SH.
        self.scale_residual_range = scale_residual_range
        self.opacity_residual_range = opacity_residual_range
        self.rotation_residual_range = rotation_residual_range
        self.sh_residual_range = sh_residual_range
        self.raw_dim = 3 + 3 + 1 + 4 + 3 * self.sh_dim

        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gaussians_per_anchor * self.raw_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _circle_offsets(
        self,
        anchors: torch.Tensor,
        spacing: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = anchors.shape
        num_views = extrinsics.shape[1]
        if n % num_views != 0:
            raise ValueError(
                "Expected anchors to be grouped evenly by source view, got "
                f"{n} anchors and {num_views} views."
            )
        anchors_per_view = n // num_views
        camera_origins = extrinsics[..., :3, 3]
        camera_origins = (
            camera_origins[:, :, None]
            .expand(b, num_views, anchors_per_view, 3)
            .reshape(b, n, 3)
        )
        ray_direction = F.normalize(
            anchors.detach() - camera_origins.detach(),
            dim=-1,
            eps=1e-6,
        )

        z_axis = torch.zeros_like(ray_direction)
        z_axis[..., 2] = 1
        y_axis = torch.zeros_like(ray_direction)
        y_axis[..., 1] = 1
        use_y_axis = ray_direction[..., 2].abs() > 0.9
        reference_axis = torch.where(use_y_axis[..., None], y_axis, z_axis)
        tangent_u = F.normalize(
            torch.cross(ray_direction, reference_axis, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        tangent_v = F.normalize(
            torch.cross(ray_direction, tangent_u, dim=-1),
            dim=-1,
            eps=1e-6,
        )

        angles = torch.arange(
            self.gaussians_per_anchor,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        angles = angles * (2 * torch.pi / self.gaussians_per_anchor)
        circle_directions = (
            tangent_u[:, :, None] * angles.cos()[None, None, :, None]
            + tangent_v[:, :, None] * angles.sin()[None, None, :, None]
        )
        radius = self.ring_radius_beta * spacing.detach()
        return radius[:, :, None] * circle_directions

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        parent_gaussians: Gaussians,
        anchor_spacing: torch.Tensor | None,
        extrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor | Gaussians]:
        b, n, _ = tokens.shape
        raw = self.head(tokens)
        raw = rearrange(
            raw,
            "b n (k c) -> b n k c",
            k=self.gaussians_per_anchor,
            c=self.raw_dim,
        )

        cursor = 0
        raw_offset = raw[..., cursor:cursor + 3]
        cursor += 3
        raw_scale = raw[..., cursor:cursor + 3]
        cursor += 3
        raw_opacity = raw[..., cursor:cursor + 1].squeeze(-1)
        cursor += 1
        raw_rotation = raw[..., cursor:cursor + 4]
        cursor += 4
        raw_sh = raw[..., cursor:cursor + 3 * self.sh_dim]

        if anchor_spacing is None:
            spacing = parent_gaussians.scales.detach().amax(dim=-1, keepdim=True)
        else:
            spacing = anchor_spacing
            while spacing.ndim < 3:
                spacing = spacing.unsqueeze(-1)
        spacing = spacing.clamp_min(1e-6)

        initial_offsets = self._circle_offsets(anchors, spacing, extrinsics)
        learned_offsets = torch.tanh(raw_offset) * (self.offset_radius * spacing[:, :, None])
        total_offsets = initial_offsets + learned_offsets
        initial_means = anchors[:, :, None] + initial_offsets
        child_means = anchors[:, :, None] + total_offsets

        parent_scales = parent_gaussians.scales[:, :, None]
        scale_delta = torch.tanh(raw_scale) * self.scale_residual_range
        child_scales = parent_scales * torch.exp(scale_delta)
        child_scales = child_scales.clamp_min(1e-6)

        parent_rotations = parent_gaussians.rotations[:, :, None]
        rotation_delta = torch.tanh(raw_rotation) * self.rotation_residual_range
        child_rotations = parent_rotations + rotation_delta
        child_rotations = F.normalize(child_rotations, dim=-1)

        parent_opacity = parent_gaussians.opacities[:, :, None].clamp(1e-6, 1 - 1e-6)
        opacity_delta = torch.tanh(raw_opacity) * self.opacity_residual_range
        child_opacities = torch.sigmoid(
            torch.logit(parent_opacity)
            + opacity_delta
        )

        parent_harmonics = parent_gaussians.harmonics[:, :, None]
        sh_delta = torch.tanh(raw_sh) * self.sh_residual_range
        child_harmonics = parent_harmonics + rearrange(
            sh_delta,
            "b n k (rgb sh) -> b n k rgb sh",
            rgb=3,
        )

        child_covariances = build_covariance(child_scales, child_rotations)
        child_gaussians = Gaussians(
            means=rearrange(child_means, "b n k xyz -> b (n k) xyz"),
            covariances=rearrange(child_covariances, "b n k i j -> b (n k) i j"),
            rotations=rearrange(child_rotations, "b n k q -> b (n k) q"),
            scales=rearrange(child_scales, "b n k xyz -> b (n k) xyz"),
            harmonics=rearrange(child_harmonics, "b n k rgb sh -> b (n k) rgb sh"),
            opacities=rearrange(child_opacities, "b n k -> b (n k)"),
        )
        return {
            "gaussians": child_gaussians,
            "means": child_means,
            "initial_means": initial_means,
            "initial_offsets": initial_offsets,
            "learned_offsets": learned_offsets,
            "offsets": total_offsets,
            "raw_opacity": raw_opacity,
            "opacities": child_opacities,
            "scales": child_scales,
        }


def scale_gaussian_scaffold(
    gaussians: Gaussians,
    scale_divisor: float = 2.4,
    opacity_multiplier: float = 0.6,
) -> Gaussians:
    scales = gaussians.scales / scale_divisor
    opacities = (gaussians.opacities * opacity_multiplier).clamp(0, 1)
    covariances = build_covariance(scales, gaussians.rotations)
    return Gaussians(
        means=gaussians.means,
        covariances=covariances,
        rotations=gaussians.rotations,
        scales=scales,
        harmonics=gaussians.harmonics,
        opacities=opacities,
    )
