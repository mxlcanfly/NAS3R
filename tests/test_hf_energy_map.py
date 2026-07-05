import torch

from src.misc.hf_energy_utils import estimate_hf_energy_sigma_sq
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
