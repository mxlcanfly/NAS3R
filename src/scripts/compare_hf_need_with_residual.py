from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize

from src.misc.hf_energy_utils import compare_hf_need_with_residual, show_hf_need_residual_comparison


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float() / 255.0


def sorted_images(root: Path) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found under {root}")
    return paths


def paired_paths(lr_root: Path, hr_root: Path) -> list[tuple[Path, Path]]:
    lr_paths = sorted_images(lr_root)
    hr_by_stem = {p.stem: p for p in sorted_images(hr_root)}
    pairs = []
    for lr_path in lr_paths:
        hr_path = hr_by_stem.get(lr_path.stem)
        if hr_path is not None:
            pairs.append((lr_path, hr_path))
    if not pairs:
        raise ValueError(f"No matching LR/HR image stems found in {lr_root} and {hr_root}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-root", type=Path, required=True)
    parser.add_argument("--hr-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--sigma-sq", type=float, default=None)
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--show-first", action="store_true")
    parser.add_argument("--save-first", type=Path, default=None)
    args = parser.parse_args()

    pairs = paired_paths(args.lr_root, args.hr_root)[: args.max_images]
    corrs = []
    first_lr = None
    first_hr = None

    for lr_path, hr_path in pairs:
        lr = load_rgb(lr_path)
        hr = load_rgb(hr_path)
        expected_hr = (lr.shape[-2] * args.scale_factor, lr.shape[-1] * args.scale_factor)
        if hr.shape[-2:] != expected_hr:
            hr = resize(hr, list(expected_hr), antialias=True)

        img_lr = lr.unsqueeze(0)
        hr_gt = hr.unsqueeze(0)
        result = compare_hf_need_with_residual(
            img_lr,
            hr_gt,
            sigma_sq=args.sigma_sq,
            ksize=args.ksize,
            scale_factor=args.scale_factor,
        )
        corr = result["spearman"].item()
        corrs.append(corr)
        print(f"{lr_path.name}: spearman={corr:.4f}")

        if first_lr is None:
            first_lr = img_lr
            first_hr = hr_gt

    corr_tensor = torch.tensor(corrs)
    print(
        "summary: "
        f"n={len(corrs)}, "
        f"mean={corr_tensor.mean().item():.4f}, "
        f"median={corr_tensor.median().item():.4f}, "
        f"min={corr_tensor.min().item():.4f}, "
        f"max={corr_tensor.max().item():.4f}"
    )

    if args.show_first or args.save_first is not None:
        assert first_lr is not None and first_hr is not None
        show_hf_need_residual_comparison(
            first_lr,
            first_hr,
            sigma_sq=args.sigma_sq,
            ksize=args.ksize,
            scale_factor=args.scale_factor,
            save_path=str(args.save_first) if args.save_first is not None else None,
        )


if __name__ == "__main__":
    main()
