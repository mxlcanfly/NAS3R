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
        hidden_dim: int = 256,
        camera_offset_beta: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.gaussians_per_anchor = gaussians_per_anchor
        self.camera_offset_beta = camera_offset_beta
        self.sh_dim = (sh_degree + 1) ** 2
        self.raw_dim = 3 + 3 + 1 + 4 + 3 * self.sh_dim

        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gaussians_per_anchor * self.raw_dim),
        )
        self._initialize_head()

    def _initialize_head(self) -> None:
        output = self.head[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        for child_idx in range(self.gaussians_per_anchor):
            offset_start = child_idx * self.raw_dim
            offset_end = offset_start + 3
            nn.init.normal_(output.weight[offset_start:offset_end], mean=0.0, std=1e-3)
            nn.init.normal_(output.bias[offset_start:offset_end], mean=0.0, std=1e-3)

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        parent_gaussians: Gaussians,
        anchor_spacing: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        image_shape: tuple[int, int] | torch.Size | None = None,
        source_view_indices: torch.Tensor | None = None,
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

        initial_child_means = anchors[:, :, None].expand(
            -1,
            -1,
            self.gaussians_per_anchor,
            -1,
        )
        if anchor_spacing is None:
            anchor_spacing = torch.ones_like(anchors[..., :1])
        camera_offset = torch.zeros_like(initial_child_means)
        if extrinsics is not None and source_view_indices is not None:
            camera_centers = extrinsics[..., :3, 3]
            source_camera_centers = torch.gather(
                camera_centers,
                dim=1,
                index=source_view_indices[..., None].expand(-1, -1, 3),
            )
            direction_to_camera = F.normalize(
                source_camera_centers - anchors,
                dim=-1,
            )
            child_factors = torch.linspace(
                1 / self.gaussians_per_anchor,
                1,
                self.gaussians_per_anchor,
                device=anchors.device,
                dtype=anchors.dtype,
            )
            camera_offset = (
                direction_to_camera[:, :, None]
                * anchor_spacing[:, :, None]
                * child_factors[None, None, :, None]
                * self.camera_offset_beta
            )
        geometry_offset = anchor_spacing[:, :, None, :] * raw_offset
        child_means = initial_child_means + camera_offset + geometry_offset

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
        }

def scale_gaussian_scaffold(
    gaussians: Gaussians,
    scale_divisor: float = 4.0,
    opacity_multiplier: float = 1.0,
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
