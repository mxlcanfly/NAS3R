from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from ...geometry.projection import (
    homogenize_points,
    sample_image_grid,
    transform_cam2world,
    unproject,
)
from .local_entropy import compute_local_shannon_entropy


@dataclass
class EntropyGuidedPoints:
    means: torch.Tensor
    coordinates: torch.Tensor
    source_view: torch.Tensor
    source_pixel_index: torch.Tensor
    entropy: torch.Tensor


class EntropyGuidedPointSampler:
    """Sample a fixed number of SR pixels per view and lift them into 3D."""

    def __init__(
        self,
        points_per_view: int,
        num_gray_levels: int = 256,
        window_size: int = 9,
        probability_eps: float = 1e-6,
    ) -> None:
        if points_per_view <= 0:
            raise ValueError("points_per_view must be positive.")
        self.points_per_view = points_per_view
        self.num_gray_levels = num_gray_levels
        self.window_size = window_size
        self.probability_eps = probability_eps

    @torch.no_grad()
    def __call__(
        self,
        sr_images: torch.Tensor,
        lr_depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        deterministic: bool = False,
    ) -> EntropyGuidedPoints:
        b, v, _, sr_h, sr_w = sr_images.shape
        if self.points_per_view > sr_h * sr_w:
            raise ValueError(
                f"Cannot sample {self.points_per_view} unique points from "
                f"an image with {sr_h * sr_w} pixels."
            )

        entropy = compute_local_shannon_entropy(
            sr_images,
            num_gray_levels=self.num_gray_levels,
            window_size=self.window_size,
        )
        probabilities = entropy.flatten(start_dim=-2) + self.probability_eps
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        probabilities_flat = rearrange(probabilities, "b v p -> (b v) p")
        if deterministic:
            generator = torch.Generator(device=probabilities_flat.device)
            generator.manual_seed(0)
            sampled_index = torch.multinomial(
                probabilities_flat,
                self.points_per_view,
                replacement=False,
                generator=generator,
            )
        else:
            sampled_index = torch.multinomial(
                probabilities_flat,
                self.points_per_view,
                replacement=False,
            )
        sampled_index = rearrange(
            sampled_index,
            "(b v) n -> b v n",
            b=b,
            v=v,
        )

        coordinate_grid, _ = sample_image_grid(
            (sr_h, sr_w),
            device=sr_images.device,
        )
        coordinate_grid = coordinate_grid.to(dtype=sr_images.dtype)
        coordinate_grid = coordinate_grid.reshape(sr_h * sr_w, 2)
        coordinates = coordinate_grid[sampled_index]

        depth_sr = F.interpolate(
            rearrange(lr_depth.detach(), "b v h w -> (b v) 1 h w"),
            size=(sr_h, sr_w),
            mode="bilinear",
            align_corners=False,
        )
        depth_sr = rearrange(
            depth_sr,
            "(b v) 1 h w -> b v (h w)",
            b=b,
            v=v,
        )
        sampled_depth = depth_sr.gather(dim=-1, index=sampled_index)

        camera_points = unproject(
            coordinates,
            sampled_depth,
            intrinsics[:, :, None],
        )
        means = transform_cam2world(
            homogenize_points(camera_points),
            extrinsics[:, :, None],
        )[..., :3]

        source_view = torch.arange(
            v,
            device=sr_images.device,
        )[None, :, None].expand(b, v, self.points_per_view)
        sampled_entropy = entropy.flatten(start_dim=-2).gather(
            dim=-1,
            index=sampled_index,
        )

        return EntropyGuidedPoints(
            means=rearrange(means, "b v n xyz -> b (v n) xyz"),
            coordinates=rearrange(coordinates, "b v n xy -> b (v n) xy"),
            source_view=rearrange(source_view, "b v n -> b (v n)"),
            source_pixel_index=rearrange(sampled_index, "b v n -> b (v n)"),
            entropy=rearrange(sampled_entropy, "b v n -> b (v n)"),
        )
