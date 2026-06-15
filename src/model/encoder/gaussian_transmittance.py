from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GaussianTransmittanceResult:
    transmittance: Tensor
    valid_mask: Tensor


class GaussianTransmittanceEstimator(nn.Module):
    """Estimate each Gaussian's transmittance in every input camera."""

    def __init__(
        self,
        target_chunk_size: int = 256,
        min_pixel_std: float = 0.3,
        depth_epsilon: float = 1e-6,
        alpha_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.target_chunk_size = target_chunk_size
        self.min_pixel_std = min_pixel_std
        self.depth_epsilon = depth_epsilon
        self.alpha_epsilon = alpha_epsilon

    def _project_gaussians(
        self,
        means: Tensor,
        covariances: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project world-space Gaussians to normalized image coordinates."""
        batch_size, num_views = extrinsics.shape[:2]
        num_gaussians = means.shape[1]
        world_to_camera = torch.linalg.inv(extrinsics)
        rotation = world_to_camera[..., :3, :3]

        means_homogeneous = torch.cat(
            [means, torch.ones_like(means[..., :1])],
            dim=-1,
        )
        means_camera = torch.einsum(
            "bvij,bnj->bvni",
            world_to_camera,
            means_homogeneous,
        )[..., :3]
        depths = means_camera[..., 2]
        safe_depths = torch.where(
            depths.abs() > self.depth_epsilon,
            depths,
            torch.full_like(depths, self.depth_epsilon),
        )

        fx = intrinsics[..., 0, 0, None]
        fy = intrinsics[..., 1, 1, None]
        cx = intrinsics[..., 0, 2, None]
        cy = intrinsics[..., 1, 2, None]
        x, y = means_camera[..., 0], means_camera[..., 1]
        projected_means = torch.stack(
            [
                fx * x / safe_depths + cx,
                fy * y / safe_depths + cy,
            ],
            dim=-1,
        )

        covariances_camera = (
            rotation[:, :, None]
            @ covariances[:, None]
            @ rotation[:, :, None].transpose(-1, -2)
        )
        jacobian = torch.zeros(
            (*means_camera.shape[:-1], 2, 3),
            device=means.device,
            dtype=means.dtype,
        )
        jacobian[..., 0, 0] = fx / safe_depths
        jacobian[..., 0, 2] = -fx * x / safe_depths.square()
        jacobian[..., 1, 1] = fy / safe_depths
        jacobian[..., 1, 2] = -fy * y / safe_depths.square()
        covariances_2d = (
            jacobian @ covariances_camera @ jacobian.transpose(-1, -2)
        )

        height, width = image_shape
        min_variance = means.new_tensor(
            [
                (self.min_pixel_std / width) ** 2,
                (self.min_pixel_std / height) ** 2,
            ]
        )
        covariances_2d = covariances_2d + torch.diag_embed(
            min_variance.expand(batch_size, num_views, num_gaussians, 2)
        )

        finite = (
            torch.isfinite(projected_means).all(dim=-1)
            & torch.isfinite(depths)
            & torch.isfinite(covariances_2d).all(dim=(-1, -2))
        )
        projectable_mask = (
            finite
            & (depths > self.depth_epsilon)
        )
        valid_mask = (
            projectable_mask
            & (projected_means[..., 0] >= 0)
            & (projected_means[..., 0] <= 1)
            & (projected_means[..., 1] >= 0)
            & (projected_means[..., 1] <= 1)
        )
        return (
            projected_means,
            covariances_2d,
            depths,
            projectable_mask,
            valid_mask,
        )

    def _compute_target_chunk(
        self,
        means_2d: Tensor,
        inverse_covariance: Tensor,
        view_depths: Tensor,
        projectable: Tensor,
        view_opacities: Tensor,
        target_means: Tensor,
        target_depths: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        # Rows are potential foreground occluders and columns are target
        # Gaussian centers.
        differences = target_means[None] - means_2d[:, None]
        transformed = torch.einsum(
            "nmi,nij->nmj",
            differences,
            inverse_covariance,
        )
        mahalanobis_squared = (transformed * differences).sum(dim=-1)
        alpha_at_target = view_opacities[:, None] * torch.exp(
            -0.5 * mahalanobis_squared
        )
        alpha_at_target = alpha_at_target.clamp(
            0,
            1 - self.alpha_epsilon,
        )

        foreground = (
            view_depths[:, None]
            < target_depths[None] - self.depth_epsilon
        )
        contributes = (
            foreground
            & projectable[:, None]
            & target_valid[None]
        )
        log_transmittance = torch.where(
            contributes,
            torch.log1p(-alpha_at_target),
            torch.zeros_like(alpha_at_target),
        ).sum(dim=0)
        transmittance = torch.exp(log_transmittance)
        return torch.where(
            target_valid,
            transmittance,
            torch.ones_like(transmittance),
        )

    @torch.no_grad()
    def forward(
        self,
        means: Tensor,
        covariances: Tensor,
        opacities: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
        source_view_indices: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> GaussianTransmittanceResult:
        """
        Return per-target, per-camera tensors with shape (B, K, V).

        Invalid target projections receive transmittance 1 and are identified
        by valid_mask=False.
        """
        (
            projected_means,
            covariances_2d,
            depths,
            projectable_mask,
            valid_mask,
        ) = (
            self._project_gaussians(
                means,
                covariances,
                extrinsics,
                intrinsics,
                image_shape,
            )
        )
        inverse_covariances_2d = torch.linalg.inv(covariances_2d)
        opacities = opacities.clamp(0, 1 - self.alpha_epsilon)
        if (
            source_view_indices is not None
            and source_view_indices.shape != means.shape[:2]
        ):
            raise ValueError(
                "source_view_indices must have shape [B, N], got "
                f"{tuple(source_view_indices.shape)}"
            )
        if target_mask is None:
            target_mask = torch.ones(
                means.shape[:2],
                dtype=torch.bool,
                device=means.device,
            )
        elif target_mask.shape != means.shape[:2]:
            raise ValueError(
                "target_mask must have shape [B, N], got "
                f"{tuple(target_mask.shape)}"
            )
        target_counts = target_mask.sum(dim=1)
        if not torch.equal(
            target_counts,
            target_counts[:1].expand_as(target_counts),
        ):
            raise ValueError(
                "Each batch item must select the same number of targets"
            )

        batch_results = []
        batch_valid_masks = []
        for batch_index in range(means.shape[0]):
            view_results = []
            view_valid_results = []
            for view_index in range(extrinsics.shape[1]):
                occluder_mask = (
                    source_view_indices[batch_index] == view_index
                    if source_view_indices is not None
                    else torch.ones(
                        means.shape[1],
                        dtype=torch.bool,
                        device=means.device,
                    )
                )
                means_2d = projected_means[
                    batch_index,
                    view_index,
                    occluder_mask,
                ]
                inverse_covariance = inverse_covariances_2d[
                    batch_index, view_index
                ][occluder_mask]
                view_depths = depths[
                    batch_index,
                    view_index,
                    occluder_mask,
                ]
                projectable = projectable_mask[
                    batch_index,
                    view_index,
                    occluder_mask,
                ]
                selected_targets = target_mask[batch_index]
                target_means_all = projected_means[
                    batch_index, view_index, selected_targets
                ]
                target_depths_all = depths[
                    batch_index, view_index, selected_targets
                ]
                target_valid_all = valid_mask[
                    batch_index, view_index, selected_targets
                ]
                view_opacities = opacities[batch_index, occluder_mask]

                target_results = []
                for start in range(
                    0,
                    target_means_all.shape[0],
                    self.target_chunk_size,
                ):
                    end = min(
                        start + self.target_chunk_size,
                        target_means_all.shape[0],
                    )
                    chunk_transmittance = self._compute_target_chunk(
                        means_2d,
                        inverse_covariance,
                        view_depths,
                        projectable,
                        view_opacities,
                        target_means_all[start:end],
                        target_depths_all[start:end],
                        target_valid_all[start:end],
                    )
                    target_results.append(chunk_transmittance)

                view_results.append(torch.cat(target_results, dim=0))
                view_valid_results.append(target_valid_all)
            batch_results.append(torch.stack(view_results, dim=-1))
            batch_valid_masks.append(
                torch.stack(view_valid_results, dim=-1)
            )

        return GaussianTransmittanceResult(
            transmittance=torch.stack(batch_results, dim=0),
            valid_mask=torch.stack(batch_valid_masks, dim=0),
        )
