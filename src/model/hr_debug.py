from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rgb_to_numpy(image: Tensor) -> np.ndarray:
    return (
        image.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _depth_limits(*depths: Tensor) -> tuple[float, float]:
    valid_values = []
    for depth in depths:
        depth = depth.detach().float()
        valid = depth[torch.isfinite(depth) & (depth > 0)]
        if valid.numel() > 0:
            valid_values.append(valid.flatten())
    if not valid_values:
        return 0.0, 1.0

    values = torch.cat(valid_values)
    vmin = values.quantile(0.01).item()
    vmax = values.quantile(0.99).item()
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_hr_debug_plane(
    lr_target: Tensor,
    bicubic_target: Tensor,
    hr_target: Tensor,
    rendered_hr: Tensor,
    refined_depth: Tensor,
    rendered_depth: Tensor,
    psnr: float,
    path: Path,
) -> None:
    vmin, vmax = _depth_limits(refined_depth, rendered_depth)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rgb_panels = (
        (lr_target, "LR Target GT"),
        (bicubic_target, "Bicubic Upsampled LR"),
        (hr_target, "HR Target GT"),
        (rendered_hr, f"Rendered HR (PSNR {psnr:.2f} dB)"),
    )
    for axis, (image, title) in zip(axes.flat[:4], rgb_panels):
        axis.imshow(_rgb_to_numpy(image))
        axis.set_title(title)
        axis.axis("off")

    depth_panels = (
        (refined_depth, "Refined HR Depth (Context 0)"),
        (rendered_depth, "Rendered HR Depth (Target 0)"),
    )
    for axis, (depth, title) in zip(axes.flat[4:], depth_panels):
        axis.imshow(
            depth.detach().float().cpu().numpy(),
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.axis("off")

    path.parent.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
