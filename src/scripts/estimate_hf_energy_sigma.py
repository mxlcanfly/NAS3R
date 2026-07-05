from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize

from src.misc.hf_energy_utils import estimate_hf_energy_sigma_sq


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_batches(root: Path, batch_size: int, image_size: int | None):
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found under {root}")

    batch = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor = pil_to_tensor(image).float() / 255.0
        if image_size is not None:
            tensor = resize(tensor, [image_size, image_size], antialias=True)
        batch.append(tensor)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch = []

    if batch:
        yield torch.stack(batch, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--one-band", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    batches = iter_image_batches(args.image_root, args.batch_size, args.image_size)
    sigma_sq = estimate_hf_energy_sigma_sq(
        batches,
        quantile=args.quantile,
        ksize=args.ksize,
        use_two_band=not args.one_band,
        max_batches=args.max_batches,
    )
    print(f"sigma_sq={sigma_sq:.12g}")


if __name__ == "__main__":
    main()
