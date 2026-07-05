from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.refiner.need_map import hf_energy_map


@torch.no_grad()
def compute_lr_hf_need_map(
    img_lr: Tensor,
    sigma_sq: float | None = None,
    ksize: int = 5,
    use_two_band: bool = True,
) -> Tensor:
    """Small utility wrapper for computing LR high-frequency need maps."""
    return hf_energy_map(
        img_lr,
        ksize=ksize,
        sigma_sq=sigma_sq,
        use_two_band=use_two_band,
        return_raw=False,
    )


@torch.no_grad()
def estimate_hf_energy_sigma_sq(
    image_batches: Iterable[Tensor],
    quantile: float = 0.90,
    ksize: int = 5,
    use_two_band: bool = True,
    max_batches: int | None = None,
) -> float:
    """Estimate a global sigma_sq from raw smoothed HF energy maps offline.

    image_batches should yield RGB tensors in [0, 1] with shape [B, 3, H, W].
    The returned quantile is intended to be passed to hf_energy_map(...,
    sigma_sq=value) during training/evaluation.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")

    raw_chunks: list[Tensor] = []
    for batch_idx, img_lr in enumerate(image_batches):
        if max_batches is not None and batch_idx >= max_batches:
            break
        raw = hf_energy_map(
            img_lr,
            ksize=ksize,
            sigma_sq=None,
            use_two_band=use_two_band,
            return_raw=True,
        )
        raw_chunks.append(raw.detach().float().flatten().cpu())

    if not raw_chunks:
        raise ValueError("image_batches did not yield any image tensors")

    values = torch.cat(raw_chunks)
    sigma_sq = torch.quantile(values, quantile).item()
    return float(max(sigma_sq, 1e-12))


@torch.no_grad()
def show_hf_energy_map(
    img_lr: Tensor,
    sigma_sq: float | None = None,
    batch_index: int = 0,
    ksize: int = 5,
    use_two_band: bool = True,
    cmap: str = "magma",
) -> Tensor:
    """Visualize an LR image and its HF energy map with matplotlib plt.show()."""
    import matplotlib.pyplot as plt

    heatmap = compute_lr_hf_need_map(
        img_lr,
        sigma_sq=sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
    )

    image_np = img_lr[batch_index].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    heat_np = heatmap[batch_index, 0].detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("LR image")
    axes[0].axis("off")
    im = axes[1].imshow(heat_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("HF energy")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    return heatmap


@torch.no_grad()
def bicubic_residual_lr_map(
    img_lr: Tensor,
    hr_gt: Tensor,
    scale_factor: int = 4,
) -> Tensor:
    """Compute the LR-domain residual map for high-frequency content missed by bicubic.

    Args:
        img_lr: LR RGB tensor in [0, 1], shape [B, 3, H, W].
        hr_gt: HR RGB tensor in [0, 1], shape [B, 3, H*scale, W*scale].
        scale_factor: Super-resolution factor. The project default is 4.

    Returns:
        Residual map with shape [B, 1, H, W].
    """
    if img_lr.ndim != 4 or hr_gt.ndim != 4:
        raise ValueError("img_lr and hr_gt must both be [B, 3, H, W] tensors")
    if img_lr.shape[:2] != hr_gt.shape[:2] or img_lr.shape[1] != 3:
        raise ValueError(f"Expected matching [B, 3], got {tuple(img_lr.shape)} and {tuple(hr_gt.shape)}")
    expected_hw = (img_lr.shape[-2] * scale_factor, img_lr.shape[-1] * scale_factor)
    if hr_gt.shape[-2:] != expected_hw:
        raise ValueError(f"Expected hr_gt spatial size {expected_hw}, got {tuple(hr_gt.shape[-2:])}")

    bicubic = F.interpolate(img_lr, scale_factor=scale_factor, mode="bicubic")
    residual = (hr_gt - bicubic).abs()
    residual = residual.mean(1, keepdim=True)
    return F.avg_pool2d(residual, scale_factor)


def _rankdata_ordinal(x: Tensor) -> Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    return ranks


@torch.no_grad()
def spearman_corr_per_image(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute per-image Spearman rank correlation for [B, 1, H, W] maps."""
    if a.shape != b.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.ndim != 4 or a.shape[1] != 1:
        raise ValueError(f"Spearman inputs must have shape [B, 1, H, W], got {tuple(a.shape)}")

    corrs = []
    for a_i, b_i in zip(a.flatten(1), b.flatten(1)):
        rank_a = _rankdata_ordinal(a_i.detach().float())
        rank_b = _rankdata_ordinal(b_i.detach().float())
        rank_a = rank_a - rank_a.mean()
        rank_b = rank_b - rank_b.mean()
        corr = (rank_a * rank_b).mean() / (rank_a.std(unbiased=False) * rank_b.std(unbiased=False) + eps)
        corrs.append(corr)
    return torch.stack(corrs)


@torch.no_grad()
def compare_hf_need_with_residual(
    img_lr: Tensor,
    hr_gt: Tensor,
    sigma_sq: float | None = None,
    ksize: int = 5,
    use_two_band: bool = True,
    scale_factor: int = 4,
) -> dict[str, Tensor]:
    """Compare HF need map against the LR-domain bicubic residual proxy."""
    need = compute_lr_hf_need_map(
        img_lr,
        sigma_sq=sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
    )
    residual_lr = bicubic_residual_lr_map(img_lr, hr_gt, scale_factor=scale_factor)
    residual_vis = residual_lr / (residual_lr.amax(dim=(2, 3), keepdim=True) + 1e-8)
    spearman = spearman_corr_per_image(need, residual_lr)
    return {
        "need": need,
        "residual_lr": residual_lr,
        "residual_vis": residual_vis,
        "spearman": spearman,
    }


@torch.no_grad()
def summarize_hf_need_residual_step(
    img_lr: Tensor,
    hr_gt: Tensor,
    sigma_sq: float | None = None,
    auto_sigma_sq: bool = True,
    sigma_quantile: float = 0.90,
    ksize: int = 5,
    use_two_band: bool = True,
    scale_factor: int = 4,
) -> dict[str, Tensor | float]:
    """Summarize need-vs-residual agreement over all images in one step.

    img_lr and hr_gt are flattened image batches with shapes [N, 3, 64, 64] and
    [N, 3, 256, 256]. If sigma_sq is None and auto_sigma_sq is true, the current
    step's raw HF energy quantile is used for the soft-compression path.
    """
    resolved_sigma_sq = sigma_sq
    if resolved_sigma_sq is None and auto_sigma_sq:
        resolved_sigma_sq = estimate_hf_energy_sigma_sq(
            [img_lr],
            quantile=sigma_quantile,
            ksize=ksize,
            use_two_band=use_two_band,
        )

    result = compare_hf_need_with_residual(
        img_lr,
        hr_gt,
        sigma_sq=resolved_sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
        scale_factor=scale_factor,
    )
    spearman = result["spearman"].detach().float()
    result.update(
        {
            "sigma_sq": float(resolved_sigma_sq) if resolved_sigma_sq is not None else None,
            "spearman_mean": spearman.mean(),
            "spearman_std": spearman.std(unbiased=False),
            "spearman_median": spearman.median(),
            "spearman_min": spearman.min(),
            "spearman_max": spearman.max(),
        }
    )
    return result


@torch.no_grad()
def show_hf_need_residual_comparison(
    img_lr: Tensor,
    hr_gt: Tensor,
    sigma_sq: float | None = None,
    batch_index: int = 0,
    ksize: int = 5,
    use_two_band: bool = True,
    scale_factor: int = 4,
    save_path: str | None = None,
    show: bool = True,
) -> dict[str, Tensor]:
    """Show LR image, need map, bicubic residual map, and their scatter plot."""
    import matplotlib.pyplot as plt

    result = compare_hf_need_with_residual(
        img_lr,
        hr_gt,
        sigma_sq=sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
        scale_factor=scale_factor,
    )

    image_np = img_lr[batch_index].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    need_np = result["need"][batch_index, 0].detach().float().cpu().numpy()
    residual_np = result["residual_vis"][batch_index, 0].detach().float().cpu().numpy()
    need_flat = result["need"][batch_index, 0].detach().float().flatten().cpu().numpy()
    residual_flat = result["residual_lr"][batch_index, 0].detach().float().flatten().cpu().numpy()
    corr = result["spearman"][batch_index].item()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("LR image")
    axes[0].axis("off")

    im_need = axes[1].imshow(need_np, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Need map")
    axes[1].axis("off")
    fig.colorbar(im_need, ax=axes[1], fraction=0.046, pad=0.04)

    im_res = axes[2].imshow(residual_np, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Bicubic residual")
    axes[2].axis("off")
    fig.colorbar(im_res, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].scatter(need_flat, residual_flat, s=3, alpha=0.25)
    axes[3].set_title(f"Spearman={corr:.3f}")
    axes[3].set_xlabel("Need")
    axes[3].set_ylabel("Residual")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return result


@torch.no_grad()
def save_hf_need_residual_batch_grid(
    img_lr: Tensor,
    result: dict[str, Tensor | float],
    max_images: int = 32,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Save/show a grid over one step: LR, need map, residual map for many images."""
    import matplotlib.pyplot as plt

    num_images = min(img_lr.shape[0], max_images)
    need = result["need"]
    residual_vis = result["residual_vis"] if "residual_vis" in result else (
        result["residual_lr"] / (result["residual_lr"].amax(dim=(2, 3), keepdim=True) + 1e-8)
    )
    spearman = result["spearman"]

    fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images), squeeze=False)
    for i in range(num_images):
        image_np = img_lr[i].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        need_np = need[i, 0].detach().float().cpu().numpy()
        residual_np = residual_vis[i, 0].detach().float().cpu().numpy()
        corr = spearman[i].item()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"LR #{i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(need_np, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("Need")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(residual_np, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title(f"Residual, rho={corr:.3f}")
        axes[i, 2].axis("off")

    title = (
        f"HF need vs bicubic residual, n={spearman.numel()}, "
        f"mean={spearman.mean().item():.3f}, std={spearman.std(unbiased=False).item():.3f}"
    )
    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
