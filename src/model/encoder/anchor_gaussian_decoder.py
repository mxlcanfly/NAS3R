import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from ..types import Gaussians
from .common.gaussians import build_covariance
from ...geometry.projection import (
    homogenize_points,
    transform_cam2world,
    transform_world2cam,
    project_camera_space,
)


class AnchorGaussianResidualDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        gaussians_per_anchor: int = 8,
        sh_degree: int = 4,
        hidden_dim: int = 256,
        **_: object,
    ) -> None:
        super().__init__()
        self.gaussians_per_anchor = gaussians_per_anchor
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
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        parent_gaussians: Gaussians,
        anchor_spacing: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        image_shape: tuple[int, int] | torch.Size | None = None,
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

        initial_child_means = self._initialize_child_means_from_sr_grid(
            anchors,
            extrinsics,
            intrinsics,
            image_shape,
            self.gaussians_per_anchor,
        )
        if anchor_spacing is None:
            anchor_spacing = torch.ones_like(anchors[..., :1])
        geometry_offset = anchor_spacing[:, :, None, :] * raw_offset
        child_means = initial_child_means + geometry_offset

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

    def _initialize_child_means_from_sr_grid(
        self,
        anchors: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        image_shape: tuple[int, int] | torch.Size | None,
        num_children: int,
    ) -> torch.Tensor:
        if extrinsics is None or intrinsics is None or image_shape is None:
            return anchors[:, :, None].expand(-1, -1, num_children, -1)

        b, num_views = extrinsics.shape[:2]
        num_anchors = anchors.shape[1]
        if num_anchors % num_views != 0:
            raise ValueError(
                "SR-grid child initialization expects anchors grouped by source "
                f"view, got {num_anchors} anchors and {num_views} views."
            )

        anchors_per_view = num_anchors // num_views
        anchors_by_view = rearrange(
            anchors,
            "b (v r) xyz -> b v r xyz",
            v=num_views,
            r=anchors_per_view,
        )

        cam_points = transform_world2cam(
            homogenize_points(anchors_by_view),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = cam_points[..., -1].clamp_min(1e-6)
        projected_xy = project_camera_space(cam_points, intrinsics[:, :, None])

        target_h, target_w = int(image_shape[-2]), int(image_shape[-1])
        pixel_scale = torch.tensor(
            (target_w, target_h),
            device=anchors.device,
            dtype=anchors.dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            torch.arange(4, device=anchors.device, dtype=anchors.dtype),
            torch.arange(4, device=anchors.device, dtype=anchors.dtype),
            indexing="ij",
        )
        grid_offsets = torch.stack((grid_x, grid_y), dim=-1).reshape(16, 2) - 1.5

        generator = None
        if not self.training:
            generator = torch.Generator(device=anchors.device)
            generator.manual_seed(0)

        if num_children <= 16:
            noise = torch.rand(
                b,
                num_views,
                anchors_per_view,
                16,
                device=anchors.device,
                generator=generator,
            )
            cell_index = noise.argsort(dim=-1)[..., :num_children]
        else:
            repeats = (num_children + 15) // 16
            cell_index = torch.arange(
                16,
                device=anchors.device,
            ).repeat(repeats)[:num_children]
            cell_index = cell_index.view(1, 1, 1, num_children).expand(
                b,
                num_views,
                anchors_per_view,
                num_children,
            )

        cell_offsets = grid_offsets[cell_index]
        jitter = torch.rand(
            cell_offsets.shape,
            device=cell_offsets.device,
            dtype=cell_offsets.dtype,
            generator=generator,
        ) - 0.5
        pixel_offsets = cell_offsets + jitter
        child_xy = projected_xy[..., None, :] + pixel_offsets / pixel_scale
        child_xy = child_xy.clamp(0, 1)

        child_xy_h = homogenize_points(child_xy)
        ray_directions = torch.einsum(
            "...ij,...j->...i",
            intrinsics[:, :, None, None].inverse(),
            child_xy_h,
        )
        child_cam_points = ray_directions * camera_depth[..., None, None]
        child_world_points = transform_cam2world(
            homogenize_points(child_cam_points),
            extrinsics[:, :, None, None],
        )[..., :-1]

        return rearrange(
            child_world_points,
            "b v r k xyz -> b (v r) k xyz",
        )


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
