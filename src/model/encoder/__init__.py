from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_nas3rm import EncoderNAS3RM, EncoderNAS3RMCfg

ENCODERS = {
    "nas3r-m": (EncoderNAS3RM, None),
}

EncoderCfg = EncoderNAS3RMCfg

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
