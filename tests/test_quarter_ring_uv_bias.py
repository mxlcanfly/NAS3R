import math

import pytest
import torch

from src.model.utils.gaussian_child_decoder import (
    build_quarter_ring_uv_bias,
    GDStyleGaussianChildDecoder,
    GDStyleGaussianChildDecoderCfg,
)


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
