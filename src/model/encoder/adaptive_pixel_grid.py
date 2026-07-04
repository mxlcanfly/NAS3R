from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class AdaptivePixelGrid:
    fine_mask: Tensor
    hit_masks: list[Tensor]
    split_masks: list[Tensor]
    leaf_masks: list[Tensor]
    extra_leaf_masks: list[Tensor]
    level_coords: list[Tensor]
    level_indices: list[Tensor]
    level_parent_lr_indices: list[Tensor]
    hit_level_coords: list[Tensor]
    hit_level_indices: list[Tensor]
    hit_level_parent_lr_indices: list[Tensor]


def _as_error_map(error_map: Tensor) -> Tensor:
    if error_map.ndim == 5:
        if error_map.shape[2] != 1:
            raise ValueError(f"Expected error_map channel dim to be 1, got {tuple(error_map.shape)}")
        error_map = error_map[:, :, 0]
    if error_map.ndim != 4:
        raise ValueError(f"Expected error_map shape [b, v, h, w] or [b, v, 1, h, w], got {tuple(error_map.shape)}")
    return error_map


def _check_resolution(error_map: Tensor, h_lr: int, w_lr: int, max_level: int) -> tuple[int, int]:
    _, _, h_hr, w_hr = error_map.shape
    rate = 2 ** max_level
    expected_h = h_lr * rate
    expected_w = w_lr * rate
    if (h_hr, w_hr) != (expected_h, expected_w):
        raise ValueError(
            f"Expected error_map resolution {(expected_h, expected_w)} for "
            f"h_lr={h_lr}, w_lr={w_lr}, max_level={max_level}, got {(h_hr, w_hr)}."
        )
    return h_hr, w_hr


def build_topk_error_mask(
    error_map: Tensor,
    topk_ratio: float | None = 0.1,
    threshold: float | None = None,
) -> Tensor:
    error_map = _as_error_map(error_map)
    if topk_ratio is None and threshold is None:
        raise ValueError("At least one of topk_ratio or threshold must be provided.")
    if topk_ratio is not None and not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    with torch.no_grad():
        score = error_map.detach()
        mask = torch.zeros_like(score, dtype=torch.bool)

        if topk_ratio is not None:
            b, v, h, w = score.shape
            flat = score.reshape(b, v, h * w)
            k = max(1, int(flat.shape[-1] * topk_ratio))
            k = min(k, flat.shape[-1])
            topk_idx = torch.topk(flat, k=k, dim=-1, largest=True, sorted=False).indices
            topk_mask = torch.zeros_like(flat, dtype=torch.bool)
            topk_mask.scatter_(dim=-1, index=topk_idx, value=True)
            mask = topk_mask.view(b, v, h, w)

        if threshold is not None:
            threshold_mask = score >= threshold
            mask = mask | threshold_mask if topk_ratio is not None else threshold_mask

        return mask


def _downsample_bool(mask: Tensor, factor: int) -> Tensor:
    if factor == 1:
        return mask
    pooled = F.max_pool2d(
        mask.flatten(0, 1).float()[:, None],
        kernel_size=factor,
        stride=factor,
    )
    return pooled[:, 0].view(mask.shape[0], mask.shape[1], mask.shape[2] // factor, mask.shape[3] // factor).bool()


def _upsample_split_mask(mask: Tensor) -> Tensor:
    return mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def _cell_center_coords(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _flatten_level_mask(mask: Tensor, level: int, h_lr: int, w_lr: int) -> tuple[Tensor, Tensor, Tensor]:
    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        empty_coords = torch.empty((0, 2), dtype=torch.float32, device=mask.device)
        empty_parent = torch.empty((0, 4), dtype=torch.long, device=mask.device)
        return empty_coords, indices, empty_parent

    _, _, h, w = mask.shape
    coords_grid = _cell_center_coords(h, w, mask.device, torch.float32)
    coords = coords_grid[indices[:, 2], indices[:, 3]]

    scale = 2 ** level
    parent_y = torch.div(indices[:, 2], scale, rounding_mode="floor").clamp(max=h_lr - 1)
    parent_x = torch.div(indices[:, 3], scale, rounding_mode="floor").clamp(max=w_lr - 1)
    parent = torch.stack((indices[:, 0], indices[:, 1], parent_y, parent_x), dim=-1)
    return coords, indices, parent


def build_adaptive_pixel_grid(
    error_map: Tensor,
    h_lr: int,
    w_lr: int,
    max_level: int = 2,
    topk_ratio: float | None = 0.1,
    threshold: float | None = None,
    include_level0_extra: bool = False,
) -> AdaptivePixelGrid:
    error_map = _as_error_map(error_map)
    _check_resolution(error_map, h_lr, w_lr, max_level)
    fine_mask = build_topk_error_mask(error_map, topk_ratio=topk_ratio, threshold=threshold)

    split_masks: list[Tensor] = []
    hit_masks: list[Tensor] = []
    leaf_masks: list[Tensor] = []
    extra_leaf_masks: list[Tensor] = []
    cell_exists: list[Tensor] = []

    b, v = fine_mask.shape[:2]
    current_exists = torch.ones((b, v, h_lr, w_lr), dtype=torch.bool, device=fine_mask.device)
    cell_exists.append(current_exists)

    for level in range(max_level):
        factor_to_fine = 2 ** (max_level - level)
        split_mask = _downsample_bool(fine_mask, factor_to_fine) & cell_exists[level]
        hit_masks.append(split_mask)
        split_masks.append(split_mask)
        leaf_masks.append(cell_exists[level] & ~split_mask)
        cell_exists.append(_upsample_split_mask(split_mask))

    hit_masks.append(fine_mask)
    leaf_masks.append(cell_exists[max_level])

    for level, leaf_mask in enumerate(leaf_masks):
        if level == 0 and not include_level0_extra:
            extra_leaf_masks.append(torch.zeros_like(leaf_mask))
        else:
            extra_leaf_masks.append(leaf_mask)

    level_coords: list[Tensor] = []
    level_indices: list[Tensor] = []
    level_parent_lr_indices: list[Tensor] = []
    for level, mask in enumerate(extra_leaf_masks):
        coords, indices, parent = _flatten_level_mask(mask, level, h_lr, w_lr)
        level_coords.append(coords)
        level_indices.append(indices)
        level_parent_lr_indices.append(parent)

    hit_level_coords: list[Tensor] = []
    hit_level_indices: list[Tensor] = []
    hit_level_parent_lr_indices: list[Tensor] = []
    for level, mask in enumerate(hit_masks):
        coords, indices, parent = _flatten_level_mask(mask, level, h_lr, w_lr)
        hit_level_coords.append(coords)
        hit_level_indices.append(indices)
        hit_level_parent_lr_indices.append(parent)

    return AdaptivePixelGrid(
        fine_mask=fine_mask,
        hit_masks=hit_masks,
        split_masks=split_masks,
        leaf_masks=leaf_masks,
        extra_leaf_masks=extra_leaf_masks,
        level_coords=level_coords,
        level_indices=level_indices,
        level_parent_lr_indices=level_parent_lr_indices,
        hit_level_coords=hit_level_coords,
        hit_level_indices=hit_level_indices,
        hit_level_parent_lr_indices=hit_level_parent_lr_indices,
    )


def gather_level_features(feature: Tensor, level_indices: Tensor) -> Tensor:
    if feature.ndim != 5:
        raise ValueError(f"Expected feature shape [b, v, c, h, w], got {tuple(feature.shape)}")
    if level_indices.numel() == 0:
        return torch.empty((0, feature.shape[2]), dtype=feature.dtype, device=feature.device)
    return feature[
        level_indices[:, 0],
        level_indices[:, 1],
        :,
        level_indices[:, 2],
        level_indices[:, 3],
    ]


def _image_to_numpy(image: Tensor):
    image = image.detach().float().clamp(0, 1).cpu()
    return rearrange(image, "c h w -> h w c").numpy()


def _map_to_numpy(value: Tensor):
    return value.detach().float().cpu().numpy()


def visualize_adaptive_pixel_grid(
    grid: AdaptivePixelGrid,
    error_map: Tensor,
    context_image_sr: Tensor | None = None,
    context_sr_render: Tensor | None = None,
    batch_idx: int = 0,
    max_views: int = 2,
) -> None:
    import matplotlib.pyplot as plt

    error_map = _as_error_map(error_map)
    num_views = min(error_map.shape[1], max_views)
    max_level = len(grid.hit_masks) - 1

    num_extra_cols = 2 if context_sr_render is not None and context_image_sr is not None else 0
    num_cols = num_extra_cols + 2 + len(grid.hit_masks)
    _, axes = plt.subplots(num_views, num_cols, figsize=(4 * num_cols, 4 * num_views), squeeze=False)

    for view_idx in range(num_views):
        col = 0
        if context_image_sr is not None and context_sr_render is not None:
            axes[view_idx, col].imshow(_image_to_numpy(context_image_sr[batch_idx, view_idx]))
            axes[view_idx, col].set_title(f"context {view_idx} sr")
            col += 1

            axes[view_idx, col].imshow(_image_to_numpy(context_sr_render[batch_idx, view_idx]))
            axes[view_idx, col].set_title("lr gaussian render")
            col += 1

        err = error_map[batch_idx, view_idx]
        axes[view_idx, col].imshow(_map_to_numpy(err), cmap="magma")
        axes[view_idx, col].set_title(f"render error {err.mean().item():.4f}")
        col += 1

        axes[view_idx, col].imshow(_map_to_numpy(grid.fine_mask[batch_idx, view_idx]), cmap="gray")
        axes[view_idx, col].set_title("fine error mask")
        col += 1

        for level, mask in enumerate(grid.hit_masks):
            axes[view_idx, col].imshow(_map_to_numpy(mask[batch_idx, view_idx]), cmap="viridis")
            axes[view_idx, col].set_title(f"grid hit L{level}/{max_level}")
            col += 1

        for ax in axes[view_idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
