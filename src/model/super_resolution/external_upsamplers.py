from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@contextmanager
def _external_python_path(root: str | Path):
    root = str(root)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass


def _drop_cached_basicsr_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "basicsr" or module_name.startswith("basicsr."):
            del sys.modules[module_name]


def _install_lightweight_basicsr_package(python_root: str | Path) -> None:
    basicsr_root = Path(python_root) / "basicsr"
    basicsr_pkg = ModuleType("basicsr")
    basicsr_pkg.__path__ = [str(basicsr_root)]
    sys.modules["basicsr"] = basicsr_pkg

    for child in ("archs", "utils"):
        child_pkg = ModuleType(f"basicsr.{child}")
        child_pkg.__path__ = [str(basicsr_root / child)]
        setattr(basicsr_pkg, child, child_pkg)
        sys.modules[f"basicsr.{child}"] = child_pkg
        if child == "utils":
            child_pkg.get_root_logger = lambda *args, **kwargs: logging.getLogger("basicsr")


def _load_module(
    module_name: str,
    module_path: str | Path,
    python_root: str | Path,
    lightweight_basicsr: bool = False,
) -> ModuleType:
    _drop_cached_basicsr_modules()
    with _external_python_path(python_root):
        if lightweight_basicsr:
            _install_lightweight_basicsr_package(python_root)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _load_checkpoint(weight_path: str | Path):
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


def _checkpoint_params(checkpoint, preferred_key: str = "params_ema"):
    if isinstance(checkpoint, dict):
        if preferred_key in checkpoint:
            return checkpoint[preferred_key]
        for key in ("params_ema", "params", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


class FrozenHATUpsampler(nn.Module):
    def __init__(
        self,
        weight_path: str | Path,
        hat_root: str | Path = "/space0/mengxl/HAT",
        upscale: int = 4,
        img_size: int = 64,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size

        module = _load_module(
            "nas3r_external_hat_arch",
            Path(hat_root) / "hat" / "archs" / "hat_arch.py",
            Path(hat_root) / "BasicSR",
            lightweight_basicsr=False,
        )
        self.model = module.HAT(
            upscale=upscale,
            in_chans=3,
            img_size=img_size,
            window_size=window_size,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        missing, unexpected = self.model.load_state_dict(
            _checkpoint_params(_load_checkpoint(weight_path)),
            strict=False,
        )
        unexpected = [key for key in unexpected if "relative_position_index" not in key]
        unexpected = [key for key in unexpected if "relative_coords_table" not in key]
        missing = [key for key in missing if "S_Adapter" not in key]
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected HAT checkpoint mismatch. Missing: {missing}; unexpected: {unexpected}"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        self.model.eval()
        *batch, c, h, w = images.shape
        images = rearrange(images, "... c h w -> (...) c h w").clamp(0, 1)
        images = _pad_for_window(images, self.window_size)
        sr = self.model(images)
        sr = sr[..., : h * self.upscale, : w * self.upscale].clamp(0, 1)
        return sr.reshape(*batch, c, h * self.upscale, w * self.upscale)


class FrozenASteISRUpsampler(nn.Module):
    def __init__(
        self,
        weight_path: str | Path,
        asteisr_root: str | Path = "/space0/mengxl/ASteISR-main",
        upscale: int = 4,
        img_size: tuple[int, int] = (64, 64),
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size

        module = _load_module(
            "nas3r_external_asteisr_arch",
            Path(asteisr_root) / "basicsr" / "archs" / "a_hat_arch.py",
            asteisr_root,
            lightweight_basicsr=True,
        )
        self.model = module.ASteISRHAT(
            upscale=upscale,
            in_chans=3,
            img_size=list(img_size),
            window_size=window_size,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            drop_path_rate=0.1,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        self.model.load_state_dict(_checkpoint_params(_load_checkpoint(weight_path)), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        self.model.eval()
        *batch, views, c, h, w = images.shape
        images = rearrange(images, "... v c h w -> (...) v c h w").clamp(0, 1)
        flat_batch, views = images.shape[:2]

        pair_indices = [(i, i + 1 if i + 1 < views else i) for i in range(0, views, 2)]
        sr_views = images.new_empty(flat_batch, views, c, h * self.upscale, w * self.upscale)
        for left, right in pair_indices:
            pair = torch.cat([images[:, left], images[:, right]], dim=1)
            pair = _pad_for_window(pair, self.window_size)
            sr_pair = self.model(pair)
            sr_pair = sr_pair[..., : h * self.upscale, : w * self.upscale].clamp(0, 1)
            sr_views[:, left] = sr_pair[:, :3]
            if right != left:
                sr_views[:, right] = sr_pair[:, 3:]

        return sr_views.reshape(*batch, views, c, h * self.upscale, w * self.upscale)


def _pad_for_window(images: Tensor, window_size: int) -> Tensor:
    _, _, h, w = images.shape
    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w
    images = torch.cat([images, torch.flip(images, [2])], 2)[:, :, : h + h_pad, :]
    images = torch.cat([images, torch.flip(images, [3])], 3)[:, :, :, : w + w_pad]
    return images


def downsample_for_sr(images: Tensor, lr_shape: tuple[int, int]) -> Tensor:
    *batch, c, h, w = images.shape
    flat = rearrange(images, "... c h w -> (...) c h w")
    lr = F.interpolate(flat, size=lr_shape, mode="bicubic", align_corners=False, antialias=True)
    return lr.reshape(*batch, c, *lr_shape).clamp(0, 1)
