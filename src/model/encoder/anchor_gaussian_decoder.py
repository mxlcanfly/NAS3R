import torch
from einops import rearrange, repeat
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

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        parent_gaussians: Gaussians,
        anchor_spacing: torch.Tensor | None,
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

        delta_offsets = raw_offset * (self.offset_radius * spacing[:, :, None])
        child_means = anchors[:, :, None] + delta_offsets

        parent_scales = parent_gaussians.scales[:, :, None]
        child_scales = parent_scales + raw_scale
        child_scales = child_scales.clamp_min(1e-6)

        parent_rotations = parent_gaussians.rotations[:, :, None]
        child_rotations = parent_rotations + raw_rotation
        child_rotations = F.normalize(child_rotations, dim=-1)

        parent_opacity = parent_gaussians.opacities[:, :, None].clamp(1e-6, 1 - 1e-6)
        child_opacities = torch.sigmoid(
            torch.logit(parent_opacity)
            + raw_opacity
        )

        parent_harmonics = parent_gaussians.harmonics[:, :, None]
        child_harmonics = parent_harmonics + rearrange(
            raw_sh,
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
            "offsets": delta_offsets,
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
