import torch

from src.misc.hf_energy_utils import (
    estimate_hf_energy_sigma_sq,
    sr_bicubic_high_frequency_residual,
    sr_residual_vs_lr_need_map,
)
from src.model.refiner.need_map import hf_energy_map


def test_flat_image_has_zero_raw_energy():
    img = torch.full((2, 3, 64, 64), 0.5)
    energy = hf_energy_map(img, return_raw=True)
    assert energy.shape == (2, 1, 64, 64)
    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-7)


def test_edge_region_has_higher_energy_than_flat_region():
    img = torch.zeros(1, 3, 64, 64)
    img[:, :, :, 32:] = 1.0
    energy = hf_energy_map(img, sigma_sq=1e-3)

    edge = energy[:, :, :, 29:35].mean()
    flat = torch.cat([energy[:, :, :, :12], energy[:, :, :, 52:]], dim=3).mean()
    assert edge > flat + 0.1
    assert energy.min() >= 0.0
    assert energy.max() <= 1.0


def test_two_band_responds_more_to_mid_frequency_texture():
    x = torch.arange(64).view(1, 1, 1, 64)
    stripe = ((x // 4) % 2).float().expand(1, 3, 64, 64)

    one_band = hf_energy_map(stripe, return_raw=True, use_two_band=False).mean()
    two_band = hf_energy_map(stripe, return_raw=True, use_two_band=True).mean()
    assert two_band > one_band


def test_debug_max_normalization_is_per_image():
    img = torch.zeros(1, 3, 64, 64)
    img[:, :, 20:44, 20:44] = 1.0
    energy = hf_energy_map(img, sigma_sq=None, return_raw=False)
    assert torch.allclose(energy.amax(dim=(2, 3)), torch.ones(1, 1), atol=1e-5)


def test_estimate_sigma_sq_from_batches_returns_positive_float():
    imgs = [
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 3, 64, 64),
    ]
    sigma_sq = estimate_hf_energy_sigma_sq(imgs, quantile=0.9)
    assert isinstance(sigma_sq, float)
    assert sigma_sq > 0.0


def test_sr_bicubic_residual_is_zero_when_sr_equals_bicubic():
    img_lr = torch.rand(2, 3, 8, 8)
    img_sr = torch.nn.functional.interpolate(
        img_lr,
        scale_factor=4,
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    result = sr_bicubic_high_frequency_residual(img_lr, img_sr)
    assert result["bicubic"].shape == (2, 3, 32, 32)
    assert result["residual_abs"].shape == (2, 1, 32, 32)
    assert torch.allclose(result["residual_abs"], torch.zeros_like(result["residual_abs"]), atol=1e-6)


def test_sr_bicubic_residual_supports_view_axis():
    img_lr = torch.rand(2, 3, 3, 8, 8)
    img_sr = torch.nn.functional.interpolate(
        img_lr.reshape(-1, 3, 8, 8),
        scale_factor=4,
        mode="bicubic",
        align_corners=False,
    ).reshape(2, 3, 3, 32, 32).clamp(0, 1)
    result = sr_bicubic_high_frequency_residual(img_lr, img_sr)
    assert result["bicubic"].shape == (2, 3, 3, 32, 32)
    assert result["residual_vis"].shape == (2, 3, 1, 32, 32)


def test_sr_residual_vs_lr_need_map_returns_lr_scale_maps():
    img_lr = torch.rand(2, 3, 8, 8)
    img_sr = torch.nn.functional.interpolate(
        img_lr,
        scale_factor=4,
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    result = sr_residual_vs_lr_need_map(img_lr, img_sr)
    assert result["need"].shape == (2, 1, 8, 8)
    assert result["residual_lr"].shape == (2, 1, 8, 8)
    assert result["spearman"].shape == (2,)
