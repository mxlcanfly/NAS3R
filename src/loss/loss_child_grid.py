from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..geometry.projection import (
    homogenize_points,
    project_camera_space,
    transform_world2cam,
)
from .loss import Loss


@dataclass
class LossChildGridCfg:
    weight: float
    apply_after_step: int
    expected_scale: int = 4


@dataclass
class LossChildGridCfgWrapper:
    child_grid: LossChildGridCfg


class LossChildGrid(Loss[LossChildGridCfg, LossChildGridCfgWrapper]):
    """Keep decoded child centers inside their parent LR pixel cell."""

    def forward(
        self,
        child_means: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        sr_image_shape: tuple[int, int],
        global_step: int,
        parent_selection: Tensor | None = None,
    ) -> Tensor:
        if global_step < self.cfg.apply_after_step:
            return child_means.new_zeros(())

        _, _, lr_height, lr_width, _, _, _, _ = child_means.shape
        sr_height, sr_width = sr_image_shape
        if (
            sr_height != lr_height * self.cfg.expected_scale
            or sr_width != lr_width * self.cfg.expected_scale
        ):
            raise ValueError(
                "Child-grid loss expected an SR/LR scale of "
                f"{self.cfg.expected_scale}, but got "
                f"{lr_height}x{lr_width} -> {sr_height}x{sr_width}."
            )

        camera_means = transform_world2cam(
            homogenize_points(child_means),
            extrinsics[:, :, None, None, None, None, None],
        )[..., :3]
        projected_xy = project_camera_space(
            camera_means,
            intrinsics[:, :, None, None, None, None, None],
        )

        dtype = projected_xy.dtype
        device = projected_xy.device
        x_index = torch.arange(lr_width, device=device, dtype=dtype)
        y_index = torch.arange(lr_height, device=device, dtype=dtype)
        cell_left = (x_index / lr_width)[None, None, None, :, None, None, None]
        cell_right = ((x_index + 1) / lr_width)[
            None, None, None, :, None, None, None
        ]
        cell_top = (y_index / lr_height)[None, None, :, None, None, None, None]
        cell_bottom = ((y_index + 1) / lr_height)[
            None, None, :, None, None, None, None
        ]

        projected_x = projected_xy[..., 0]
        projected_y = projected_xy[..., 1]
        outside_x = F.relu(cell_left - projected_x) + F.relu(
            projected_x - cell_right
        )
        outside_y = F.relu(cell_top - projected_y) + F.relu(
            projected_y - cell_bottom
        )

        outside_x_pixels = outside_x * sr_width
        outside_y_pixels = outside_y * sr_height
        zero = torch.zeros((), device=device, dtype=dtype)
        per_child_loss = F.smooth_l1_loss(
            outside_x_pixels,
            zero.expand_as(outside_x_pixels),
            reduction="none",
        ) + F.smooth_l1_loss(
            outside_y_pixels,
            zero.expand_as(outside_y_pixels),
            reduction="none",
        )
        if parent_selection is not None:
            selected = parent_selection[..., None].expand_as(per_child_loss)
            if not selected.any():
                return child_means.new_zeros(())
            loss = per_child_loss[selected].mean()
        else:
            loss = per_child_loss.mean()
        return self.cfg.weight * loss
