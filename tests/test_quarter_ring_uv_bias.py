import math

import pytest
import torch

from src.model.utils.gaussian_child_decoder import (
    build_quarter_ring_uv_bias,
    GDStyleGaussianChildDecoder,
    GDStyleGaussianChildDecoderCfg,
)
from src.model.types import Gaussians


@pytest.mark.parametrize("num_children", [4, 8, 10, 16, 20])
def test_quarter_ring_uv_bias_stays_inside_pixel(num_children: int) -> None:
    bias = build_quarter_ring_uv_bias(num_children)
    offsets = torch.sigmoid(bias)

    assert offsets.shape == (num_children, 2)
    assert torch.isfinite(offsets).all()
    assert (offsets > 0).all()
    assert (offsets < 1).all()


@pytest.mark.parametrize("num_children", [4, 8, 10, 16, 20])
def test_quarter_ring_count_distribution(num_children: int) -> None:
    num_rings = max(1, math.ceil(math.sqrt(num_children)))
    offsets = build_quarter_ring_uv_bias(num_children, return_pre_sigmoid=False)
    radii = offsets.square().sum(dim=-1).sqrt()
    counts = [
        int(torch.isclose(radii, radius, atol=1e-6, rtol=0.0).sum())
        for radius in torch.unique(radii.round(decimals=6))
    ]

    assert len(counts) == num_rings
    assert sum(counts) == num_children
    assert max(counts) - min(counts) <= 1


def test_pre_sigmoid_bias_recovers_template() -> None:
    bias = build_quarter_ring_uv_bias(10)
    template = build_quarter_ring_uv_bias(10, return_pre_sigmoid=False)

    torch.testing.assert_close(torch.sigmoid(bias), template)


def test_decoder_initializes_offset_head_bias_from_template() -> None:
    cfg = GDStyleGaussianChildDecoderCfg(
        input_dim=8,
        hidden_dim=16,
        child_feat_dim=8,
        num_children=10,
    )
    decoder = GDStyleGaussianChildDecoder(cfg, sh_degree=0)
    last = decoder.delta_x[-1]
    expected_bias = build_quarter_ring_uv_bias(10)
    initialized_bias = last.bias.reshape(10, 3)

    assert torch.count_nonzero(last.weight) == 0
    torch.testing.assert_close(initialized_bias[:, :2], expected_bias)
    torch.testing.assert_close(initialized_bias[:, 2], torch.zeros(10))
    torch.testing.assert_close(decoder.initial_local_offset_uv(), torch.sigmoid(expected_bias))
    assert "uv_bias" not in dict(decoder.named_buffers())


def test_decoder_caps_child_scales() -> None:
    cfg = GDStyleGaussianChildDecoderCfg(
        input_dim=8,
        hidden_dim=16,
        child_feat_dim=8,
        num_children=4,
        n_frequencies=0,
        scale_max=0.3,
    )
    decoder = GDStyleGaussianChildDecoder(cfg, sh_degree=0)
    with torch.no_grad():
        decoder.attr_head[-1].bias[:3].fill_(10.0)

    gaussians = Gaussians(
        means=torch.tensor([[[0.0, 0.0, 1.0]]]),
        covariances=torch.eye(3).reshape(1, 1, 3, 3),
        rotations=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
        scales=torch.ones(1, 1, 3),
        harmonics=torch.zeros(1, 1, 3, 1),
        opacities=torch.full((1, 1), 0.5),
    )
    result = decoder(
        points=gaussians.means,
        features=torch.zeros(1, 1, cfg.input_dim),
        gaussians=gaussians,
        parent_uv=torch.zeros(1, 1, 2),
        parent_depths=torch.ones(1, 1),
        extrinsics=torch.eye(4).reshape(1, 1, 4, 4),
        intrinsics=torch.eye(3).reshape(1, 1, 3, 3),
        image_shape=(1, 1),
        points_per_view=1,
    )

    child_scales = result["gaussians"].scales
    assert child_scales.shape == (1, cfg.num_children, 3)
    assert child_scales.amax() <= cfg.scale_max
    assert child_scales.amin() >= cfg.scale_min
