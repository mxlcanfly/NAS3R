from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _binomial_kernel_5x5(dtype: torch.dtype, device: torch.device) -> Tensor:
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=dtype, device=device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d) / 256.0
    return kernel_2d.view(1, 1, 5, 5)


@torch.no_grad()
def hf_energy_map(
    img_lr: Tensor,
    ksize: int = 5,
    sigma_sq: float | None = None,
    use_two_band: bool = True,
    return_raw: bool = False,
) -> Tensor:
    """Compute a training-free high-frequency energy heatmap for LR RGB images.

    Args:
        img_lr: RGB image tensor in [0, 1] with shape [B, 3, H, W].
        ksize: Odd local averaging window used to smooth Laplacian energy.
        sigma_sq: Global normalization constant from offline statistics. When
            provided, the returned map is comparable across images/scenes.
        use_two_band: If true, use lap1**2 + 0.5 * lap2**2. Otherwise only use
            the finest band lap1**2.
        return_raw: If true, return the smoothed raw energy before normalization.

    Returns:
        Tensor with shape [B, 1, H, W].

    Note:
        When sigma_sq is None and return_raw is false, this falls back to
        per-image max normalization for debugging/visualization only. Do not use
        that path for training because it can amplify flat-image noise to 1.0 and
        breaks cross-image comparability.
    """
    if img_lr.ndim != 4 or img_lr.shape[1] != 3:
        raise ValueError(f"img_lr must have shape [B, 3, H, W], got {tuple(img_lr.shape)}")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}")

    weights = img_lr.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = (img_lr * weights).sum(dim=1, keepdim=True)

    kernel = _binomial_kernel_5x5(dtype=img_lr.dtype, device=img_lr.device)
    blur1 = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), kernel)
    lap1 = gray - blur1

    if use_two_band:
        blur2 = F.conv2d(F.pad(blur1, (2, 2, 2, 2), mode="reflect"), kernel)
        lap2 = blur1 - blur2
        energy = lap1.pow(2) + 0.5 * lap2.pow(2)
    else:
        energy = lap1.pow(2)

    pad = ksize // 2
    energy = F.avg_pool2d(
        F.pad(energy, (pad, pad, pad, pad), mode="reflect"),
        ksize,
        stride=1,
        padding=0,
    )

    if return_raw:
        return energy

    if sigma_sq is not None:
        sigma = img_lr.new_tensor(float(sigma_sq))
        return (1.0 - torch.exp(-energy / (sigma + 1e-8))).clamp(0.0, 1.0)

    denom = energy.amax(dim=(2, 3), keepdim=True) + 1e-8
    return energy / denom
