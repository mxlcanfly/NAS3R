from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn

from .encoder_nas3rm import EncoderNAS3RM, EncoderNAS3RMCfg


class FrozenNAS3RMDepthTeacher(nn.Module):
    """Frozen original NAS3R-M backbone and DPT heads for HR depth supervision."""

    def __init__(self, cfg: EncoderNAS3RMCfg, checkpoint_path: str | Path) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"NAS3R-M depth teacher checkpoint not found: {checkpoint_path}")

        teacher = EncoderNAS3RM(deepcopy(cfg))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(f"No encoder.* weights found in teacher checkpoint: {checkpoint_path}")
        missing_keys, _ = teacher.load_state_dict(encoder_state, strict=False)
        critical_prefixes = ("backbone.", "downstream_depth_head1.", "downstream_depth_head2.")
        missing_critical = [key for key in missing_keys if key.startswith(critical_prefixes)]
        if missing_critical:
            raise RuntimeError(
                "Teacher checkpoint is missing backbone/depth weights: "
                + ", ".join(missing_critical[:10])
            )

        # The teacher forward below only needs these modules. Releasing the others
        # avoids retaining a duplicate Gaussian/child/SwinIR network in memory.
        keep_prefixes = ("backbone", "downstream_depth_head1", "downstream_depth_head2", "depth_head1", "depth_head2")
        for name in list(teacher._modules):
            if not name.startswith(keep_prefixes):
                setattr(teacher, name, None)

        teacher.cfg.freeze_original_lr_network = False
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher

    def train(self, mode: bool = True):
        super().train(False)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor, intrinsics: Tensor) -> Tensor:
        self.teacher.eval()
        return self.teacher.predict_dpt_depth(images, intrinsics).detach()
