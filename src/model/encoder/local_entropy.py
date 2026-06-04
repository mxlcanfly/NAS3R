import torch
import torch.nn.functional as F


def compute_local_shannon_entropy(
    images: torch.Tensor,
    num_gray_levels: int = 256,
    window_size: int = 9,
) -> torch.Tensor:
    """Compute a local grayscale Shannon entropy map in bits."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(
            f"Expected images with shape [B, V, 3, H, W], got {tuple(images.shape)}."
        )
    if num_gray_levels <= 1:
        raise ValueError(
            f"num_gray_levels must be greater than 1, got {num_gray_levels}."
        )
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}."
        )

    images = images.detach().float()
    if images.amin().item() < 0:
        images = images * 0.5 + 0.5
    images = images.clamp(0, 1)

    grayscale = (
        images[:, :, 0] * 0.299
        + images[:, :, 1] * 0.587
        + images[:, :, 2] * 0.114
    )
    quantized = torch.clamp(
        (grayscale * num_gray_levels).long(),
        max=num_gray_levels - 1,
    )
    quantized = quantized.flatten(0, 1).unsqueeze(1)

    entropy = torch.zeros_like(quantized, dtype=torch.float32)
    padding = window_size // 2
    for gray_level in range(num_gray_levels):
        probability = F.avg_pool2d(
            (quantized == gray_level).float(),
            kernel_size=window_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        entropy -= torch.where(
            probability > 0,
            probability * probability.log2(),
            torch.zeros_like(probability),
        )

    return entropy.reshape(*images.shape[:2], *images.shape[-2:])
