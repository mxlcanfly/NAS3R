from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_nas3r import EncoderNAS3RCfg, EncoderNAS3R
from .encoder_nas3rm import EncoderNAS3RM, EncoderNAS3RMCfg
from .adaptive_pixel_grid import (
    AdaptivePixelGrid,
    build_adaptive_pixel_grid,
    build_topk_error_mask,
    gather_level_features,
    visualize_adaptive_pixel_grid,
)

ENCODERS = {

    "nas3r": (EncoderNAS3R, None),
    "nas3r-m": (EncoderNAS3RM, None),
}

EncoderCfg = EncoderNAS3RCfg | EncoderNAS3RMCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
