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
        image_feature_dim: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        self.gaussians_per_anchor = gaussians_per_anchor
        self.image_feature_dim = image_feature_dim
        self.sh_dim = (sh_degree + 1) ** 2
        self.attribute_dim = 3 + 1 + 4 + 3 * self.sh_dim

        attribute_input_dim = token_dim + image_feature_dim
        self.offset_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gaussians_per_anchor * 3),
        )
        self.attribute_head = nn.Sequential(
            nn.LayerNorm(attribute_input_dim),
            nn.Linear(attribute_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                gaussians_per_anchor * self.attribute_dim,
            ),
        )
        self._initialize_head()

    def _initialize_head(self) -> None:
        nn.init.normal_(self.offset_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.offset_head[-1].bias, mean=0.0, std=1e-3)
        nn.init.zeros_(self.attribute_head[-1].weight)
        nn.init.zeros_(self.attribute_head[-1].bias)
        # attribute layout per child: scale(3) | opacity(1) | rotation(4) | sh(...)
        #
        # opacity: bias = -2.0 so children start at sigmoid(logit(0.5) - 2) ≈ 0.12
        #          instead of inheriting parent opacity (~0.5).  With K=8 co-located
        #          children, parent opacity 0.5 gives combined alpha ≈ 0.998; 0.12
        #          reduces this to a manageable ~0.67.
        with torch.no_grad():
            for k in range(self.gaussians_per_anchor):
                self.attribute_head[-1].bias[k * self.attribute_dim + 3] = -2.0

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        parent_gaussians: Gaussians,
        anchor_spacing: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Gaussians]:
        b, n, _ = tokens.shape
        if image_features is None:
            image_features = tokens.new_zeros(b, n, self.image_feature_dim)
        if image_features.shape != (b, n, self.image_feature_dim):
            raise ValueError(
                "Expected image_features with shape "
                f"{(b, n, self.image_feature_dim)}, got "
                f"{tuple(image_features.shape)}."
            )
        attribute_tokens = torch.cat(
            (
                tokens,
                image_features,
            ),
            dim=-1,
        )
        raw_offset = rearrange(
            self.offset_head(tokens),
            "b n (k xyz) -> b n k xyz",
            k=self.gaussians_per_anchor,
            xyz=3,
        )
        raw_attributes = rearrange(
            self.attribute_head(attribute_tokens),
            "b n (k attributes) -> b n k attributes",
            k=self.gaussians_per_anchor,
            attributes=self.attribute_dim,
        )

        cursor = 0
        raw_scale = raw_attributes[..., cursor:cursor + 3]
        cursor += 3
        raw_opacity = raw_attributes[..., cursor:cursor + 1].squeeze(-1)
        cursor += 1
        raw_rotation = raw_attributes[..., cursor:cursor + 4]
        cursor += 4
        raw_sh = raw_attributes[..., cursor:cursor + 3 * self.sh_dim]

        if anchor_spacing is None:
            anchor_spacing = torch.ones_like(anchors[..., :1])
        geometry_offset = anchor_spacing[:, :, None, :] * raw_offset
        child_means = anchors[:, :, None] + geometry_offset

        # Detach parent attributes so SR-branch gradients don't flow back into
        # the LR backbone — mirrors the resplat design (all prev_* are detached).
        parent_rotations = parent_gaussians.rotations.detach()[:, :, None]
        parent_scales = parent_gaussians.scales.detach()[:, :, None]

        child_scales = (parent_scales + raw_scale).clamp(1e-6, 0.3)

        child_rotations = parent_rotations + raw_rotation
        child_rotations = F.normalize(child_rotations, dim=-1)

        parent_opacity = parent_gaussians.opacities.detach()[:, :, None].clamp(1e-6, 1 - 1e-6)
        child_opacities = torch.sigmoid(
            torch.logit(parent_opacity)
            + raw_opacity
        )

        parent_harmonics = parent_gaussians.harmonics.detach()[:, :, None]
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
            "offsets": raw_offset,
            "world_offsets": geometry_offset,
        }

def scale_gaussian_scaffold(
    gaussians: Gaussians,
    scale_divisor: float = 4.0,
    opacity_multiplier: float = 1.0,
    preserve_original: bool = False,
) -> Gaussians | tuple[Gaussians, Gaussians]:
    """Scale down Gaussian scaffold while optionally preserving the original values.

    The original Gaussian scales and rotations can be used later to constrain
    offsets or rotation updates before the scaffold is reduced.
    """
    original_gaussians = gaussians
    scales = gaussians.scales / scale_divisor
    opacities = (gaussians.opacities * opacity_multiplier).clamp(0, 1)
    covariances = build_covariance(scales, gaussians.rotations)
    scaled_gaussians = Gaussians(
        means=gaussians.means,
        covariances=covariances,
        rotations=gaussians.rotations,
        scales=scales,
        harmonics=gaussians.harmonics,
        opacities=opacities,
    )
    if preserve_original:
        return scaled_gaussians, original_gaussians
    return scaled_gaussians
