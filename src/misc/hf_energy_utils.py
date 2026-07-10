from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


def _hf_energy_map(*args, **kwargs) -> Tensor:
    from src.model.refiner.need_map import hf_energy_map

    return hf_energy_map(*args, **kwargs)


@torch.no_grad()
def compute_lr_hf_need_map(
    img_lr: Tensor,
    sigma_sq: float | None = None,
    ksize: int = 5,
    use_two_band: bool = True,
) -> Tensor:
    """Small utility wrapper for computing LR high-frequency need maps."""
    return _hf_energy_map(
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
        raw = _hf_energy_map(
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


@torch.no_grad()
def sr_bicubic_high_frequency_residual(
    img_lr: Tensor,
    img_sr: Tensor,
    scale_factor: int = 4,
    normalize: bool = True,
) -> dict[str, Tensor]:
    """Compare SwinIR/SR output against bicubic upsampling.

    The residual ``img_sr - bicubic(img_lr)`` is a useful proxy for where the SR
    network adds detail beyond deterministic interpolation. Inputs may be shaped
    as [B, 3, H, W] or with extra leading axes such as [B, V, 3, H, W].
    """
    if img_lr.ndim < 4 or img_sr.ndim != img_lr.ndim:
        raise ValueError(f"Expected matching image tensors, got {tuple(img_lr.shape)} and {tuple(img_sr.shape)}")
    if img_lr.shape[:-3] != img_sr.shape[:-3] or img_lr.shape[-3] != 3 or img_sr.shape[-3] != 3:
        raise ValueError(f"Expected matching leading axes and RGB channels, got {tuple(img_lr.shape)} and {tuple(img_sr.shape)}")

    expected_hw = (img_lr.shape[-2] * scale_factor, img_lr.shape[-1] * scale_factor)
    if img_sr.shape[-2:] != expected_hw:
        raise ValueError(f"Expected SR spatial size {expected_hw}, got {tuple(img_sr.shape[-2:])}")

    leading_shape = img_lr.shape[:-3]
    lr_flat = img_lr.reshape(-1, *img_lr.shape[-3:]).float().clamp(0, 1)
    sr_flat = img_sr.reshape(-1, *img_sr.shape[-3:]).float().clamp(0, 1)
    bicubic_flat = F.interpolate(
        lr_flat,
        scale_factor=scale_factor,
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)

    residual_flat = sr_flat - bicubic_flat
    residual_abs_flat = residual_flat.abs().mean(dim=1, keepdim=True)
    if normalize:
        denom = residual_abs_flat.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        residual_vis_flat = residual_abs_flat / denom
    else:
        residual_vis_flat = residual_abs_flat

    return {
        "bicubic": bicubic_flat.reshape(*leading_shape, *bicubic_flat.shape[-3:]),
        "residual": residual_flat.reshape(*leading_shape, *residual_flat.shape[-3:]),
        "residual_abs": residual_abs_flat.reshape(*leading_shape, *residual_abs_flat.shape[-3:]),
        "residual_vis": residual_vis_flat.reshape(*leading_shape, *residual_vis_flat.shape[-3:]),
    }


@torch.no_grad()
def sr_residual_lr_map(
    img_lr: Tensor,
    img_sr: Tensor,
    scale_factor: int = 4,
) -> dict[str, Tensor]:
    """Collapse SR-vs-bicubic residual to the LR grid without computing need maps."""
    residual_result = sr_bicubic_high_frequency_residual(
        img_lr,
        img_sr,
        scale_factor=scale_factor,
        normalize=False,
    )
    residual_abs = residual_result["residual_abs"]
    leading_shape = residual_abs.shape[:-3]
    residual_flat = residual_abs.reshape(-1, *residual_abs.shape[-3:])
    residual_unshuffle = F.pixel_unshuffle(residual_flat, downscale_factor=scale_factor)
    residual_lr_flat = residual_unshuffle.mean(dim=1, keepdim=True)
    residual_lr_vis_flat = residual_lr_flat / residual_lr_flat.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return {
        **residual_result,
        "residual_lr": residual_lr_flat.reshape(*leading_shape, *residual_lr_flat.shape[-3:]),
        "residual_lr_vis": residual_lr_vis_flat.reshape(*leading_shape, *residual_lr_vis_flat.shape[-3:]),
    }


@torch.no_grad()
def show_sr_bicubic_high_frequency_residual(
    img_lr: Tensor,
    img_sr: Tensor,
    scale_factor: int = 4,
    batch_index: int = 0,
    view_index: int | None = None,
    residual_gain: float = 4.0,
    cmap: str = "magma",
    show: bool = True,
    save_path: str | None = None,
) -> dict[str, Tensor]:
    """Visualize where SR adds information relative to bicubic upsampling."""
    import matplotlib.pyplot as plt

    result = sr_bicubic_high_frequency_residual(
        img_lr,
        img_sr,
        scale_factor=scale_factor,
        normalize=True,
    )

    if img_lr.ndim == 5:
        if view_index is None:
            view_index = 0
        lr_i = img_lr[batch_index, view_index]
        sr_i = img_sr[batch_index, view_index]
        bicubic_i = result["bicubic"][batch_index, view_index]
        residual_i = result["residual"][batch_index, view_index]
        residual_vis_i = result["residual_vis"][batch_index, view_index, 0]
    elif img_lr.ndim == 4:
        lr_i = img_lr[batch_index]
        sr_i = img_sr[batch_index]
        bicubic_i = result["bicubic"][batch_index]
        residual_i = result["residual"][batch_index]
        residual_vis_i = result["residual_vis"][batch_index, 0]
    else:
        raise ValueError(f"Visualization supports [B,3,H,W] or [B,V,3,H,W], got {tuple(img_lr.shape)}")

    signed_vis = (0.5 + residual_i * residual_gain).clamp(0, 1)

    lr_np = lr_i.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    bicubic_np = bicubic_i.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    sr_np = sr_i.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    residual_np = residual_vis_i.detach().float().cpu().numpy()
    signed_np = signed_vis.detach().float().permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(lr_np)
    axes[0].set_title("LR")
    axes[0].axis("off")

    axes[1].imshow(bicubic_np)
    axes[1].set_title("Bicubic")
    axes[1].axis("off")

    axes[2].imshow(sr_np)
    axes[2].set_title("SR")
    axes[2].axis("off")

    im = axes[3].imshow(residual_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[3].set_title("|SR - Bicubic|")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(signed_np)
    axes[4].set_title(f"Signed residual x{residual_gain:g}")
    axes[4].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return result


@torch.no_grad()
def sr_residual_vs_lr_need_map(
    img_lr: Tensor,
    img_sr: Tensor,
    scale_factor: int = 4,
    sigma_sq: float | None = None,
    ksize: int = 5,
    use_two_band: bool = True,
    compute_spearman: bool = False,
) -> dict[str, Tensor]:
    """Compare SR-vs-bicubic residual, collapsed to LR, with LR HF need map."""
    residual_result = sr_bicubic_high_frequency_residual(
        img_lr,
        img_sr,
        scale_factor=scale_factor,
        normalize=False,
    )
    residual_abs = residual_result["residual_abs"]
    leading_shape = residual_abs.shape[:-3]
    residual_flat = residual_abs.reshape(-1, *residual_abs.shape[-3:])

    residual_unshuffle = F.pixel_unshuffle(residual_flat, downscale_factor=scale_factor)
    residual_lr_flat = residual_unshuffle.mean(dim=1, keepdim=True)
    residual_lr_vis_flat = residual_lr_flat / residual_lr_flat.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    img_lr_flat = img_lr.reshape(-1, *img_lr.shape[-3:]).float().clamp(0, 1)
    need_flat = compute_lr_hf_need_map(
        img_lr_flat,
        sigma_sq=sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
    )

    result = {
        **residual_result,
        "residual_lr": residual_lr_flat.reshape(*leading_shape, *residual_lr_flat.shape[-3:]),
        "residual_lr_vis": residual_lr_vis_flat.reshape(*leading_shape, *residual_lr_vis_flat.shape[-3:]),
        "need": need_flat.reshape(*leading_shape, *need_flat.shape[-3:]),
    }
    if compute_spearman:
        spearman = spearman_corr_per_image(need_flat, residual_lr_flat)
        result["spearman"] = spearman.reshape(*leading_shape)
    return result


@torch.no_grad()
def show_sr_residual_vs_lr_need_map(
    img_lr: Tensor,
    img_sr: Tensor,
    scale_factor: int = 4,
    sigma_sq: float | None = None,
    ksize: int = 5,
    use_two_band: bool = True,
    batch_index: int = 0,
    view_index: int | None = None,
    cmap: str = "magma",
    show: bool = True,
    save_path: str | None = None,
) -> dict[str, Tensor]:
    """Visualize LR HF need map against pixel-unshuffled SR residual at LR scale."""
    import matplotlib.pyplot as plt

    result = sr_residual_vs_lr_need_map(
        img_lr,
        img_sr,
        scale_factor=scale_factor,
        sigma_sq=sigma_sq,
        ksize=ksize,
        use_two_band=use_two_band,
        compute_spearman=True,
    )

    if img_lr.ndim == 5:
        if view_index is None:
            view_index = 0
        need_i = result["need"][batch_index, view_index, 0]
        residual_i = result["residual_lr_vis"][batch_index, view_index, 0]
    elif img_lr.ndim == 4:
        need_i = result["need"][batch_index, 0]
        residual_i = result["residual_lr_vis"][batch_index, 0]
    else:
        raise ValueError(f"Visualization supports [B,3,H,W] or [B,V,3,H,W], got {tuple(img_lr.shape)}")

    need_np = need_i.detach().float().cpu().numpy()
    residual_np = residual_i.detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im_need = axes[0].imshow(need_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("LR need map")
    axes[0].axis("off")
    fig.colorbar(im_need, ax=axes[0], fraction=0.046, pad=0.04)

    im_res = axes[1].imshow(residual_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("SR residual -> LR")
    axes[1].axis("off")
    fig.colorbar(im_res, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return result


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
