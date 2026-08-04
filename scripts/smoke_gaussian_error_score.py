#!/usr/bin/env python3
"""Smoke-test full 256-D scene-level Gaussian error lifting and refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.global_cfg import set_cfg
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.encoder.common.gaussians import build_covariance


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def psnr(ground_truth: Tensor, prediction: Tensor) -> Tensor:
    mse = (ground_truth.float() - prediction.float()).square().mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def max_gaussian_difference(first: Any, second: Any) -> dict[str, float]:
    return {
        name: float(
            (getattr(first, name).float() - getattr(second, name).float())
            .abs()
            .max()
        )
        for name in (
            "means",
            "covariances",
            "rotations",
            "scales",
            "harmonics",
            "opacities",
        )
    }


def relative_attribution_error(
    per_view_numerator: Tensor,
    pixel_error: Tensor,
    rendered_alpha: Tensor,
) -> Tensor:
    gaussian_sum = per_view_numerator.sum(dim=2)
    image_sum = (pixel_error * rendered_alpha).sum(dim=(-1, -2))
    absolute_mass = (pixel_error.abs() * rendered_alpha).sum(dim=(-1, -2))
    return (gaussian_sum - image_sum).abs() / absolute_mass.clamp_min(1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained_weights/re10k_nas3r-m_pretrained-I.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gaussian_256d_refiner_smoke"),
    )
    parser.add_argument(
        "--skip-zero-lifting",
        action="store_true",
        help="Skip the additional eight zero-feature probe passes.",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        raw_cfg = compose(
            config_name="main",
            overrides=[
                "+experiment=nas3r-m/pretrained/re10k-I",
                "mode=test",
                "model.encoder.refine_enabled=true",
            ],
        )
    set_cfg(raw_cfg)
    cfg = load_typed_root_config(raw_cfg)
    cfg.data_loader.val.batch_size = 1
    cfg.data_loader.val.num_workers = 0
    cfg.data_loader.val.persistent_workers = False

    encoder, _ = get_encoder(cfg.model.encoder)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    missing = [key for key in missing if not key.startswith("refiner.")]
    unexpected = [key for key in unexpected if not key.startswith("refiner.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    checkpoint_step = int(checkpoint.get("global_step", 0))
    del checkpoint, encoder_state

    encoder = encoder.to(device).eval()
    decoder = get_decoder(cfg.model.decoder).to(device).eval()
    tracker = StepTracker()
    tracker.set_step(checkpoint_step)
    dataset = get_dataset(cfg.dataset, "val", tracker)[0]
    batch = move_to_device(
        next(iter(DataLoader(dataset, batch_size=1, num_workers=0))),
        device,
    )
    context = batch["context"]
    _, views, _, height, width = context["image"].shape

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoder_output = encoder(context, global_step=checkpoint_step, target=None)
    gaussians = encoder_output["gaussians"]
    extrinsics = (
        encoder_output["extrinsics"]["c"]
        if encoder.cfg.estimating_pose
        else context["extrinsics"]
    ).float()
    intrinsics = (
        encoder_output["intrinsics"]["c"]
        if encoder.cfg.estimating_focal
        else context["intrinsics"]
    ).float()

    with torch.no_grad():
        initial_render = decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            context["near"],
            context["far"],
            (height, width),
        ).color
    pixel_feature_error, pixel_rgb_error = encoder.refiner.build_pixel_render_error(
        initial_render, context["image"]
    )
    expected_feature_shape = (1, views, 256, height, width)
    expected_rgb_shape = (1, views, 3, height, width)
    if tuple(pixel_feature_error.shape) != expected_feature_shape:
        raise RuntimeError(
            f"Feature error shape mismatch: expected={expected_feature_shape}, "
            f"got={tuple(pixel_feature_error.shape)}"
        )
    if tuple(pixel_rgb_error.shape) != expected_rgb_shape:
        raise RuntimeError(
            f"RGB error shape mismatch: expected={expected_rgb_shape}, "
            f"got={tuple(pixel_rgb_error.shape)}"
        )

    lifting_cfg = encoder.cfg.error_lifting
    grads_before = sum(parameter.grad is not None for parameter in encoder.parameters())
    result = decoder.compute_gaussian_error_cues(
        gaussians,
        extrinsics,
        intrinsics,
        context["near"],
        context["far"],
        (height, width),
        feature_error=pixel_feature_error,
        rgb_error=pixel_rgb_error,
        chunk_size=lifting_cfg.chunk_size,
        max_probe_channels=lifting_cfg.max_probe_channels,
        eps=lifting_cfg.contribution_eps,
    )
    grads_after = sum(parameter.grad is not None for parameter in encoder.parameters())

    gaussian_count = gaussians.means.shape[1]
    if tuple(result["feature_cue"].shape) != (1, gaussian_count, 256):
        raise RuntimeError(f"Wrong feature cue shape: {result['feature_cue'].shape}")
    if tuple(result["rgb_cue"].shape) != (1, gaussian_count, 3):
        raise RuntimeError(f"Wrong RGB cue shape: {result['rgb_cue'].shape}")
    if int(result["feature_probe_count"]) != 8:
        raise RuntimeError(f"Expected 8 feature probes, got {result['feature_probe_count']}")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            pixel_feature_error,
            pixel_rgb_error,
            result["feature_cue"],
            result["rgb_cue"],
            result["total"],
        )
    ):
        raise RuntimeError("Pixel/Gaussian error lifting contains NaN or Inf")

    fused_cue = encoder.refiner.fuse_observation_cue(
        result["feature_cue"], result["rgb_cue"]
    ).detach()
    if tuple(fused_cue.shape) != (1, gaussian_count, 256):
        raise RuntimeError(f"Wrong fused cue shape: {fused_cue.shape}")
    zero_fused = encoder.refiner.fuse_observation_cue(
        torch.zeros_like(result["feature_cue"]),
        torch.zeros_like(result["rgb_cue"]),
    )
    if float(zero_fused.abs().max()) != 0:
        raise RuntimeError("Zero feature/RGB errors must produce a zero fused cue")

    contribution_sum = result["per_view"].sum(dim=-1)
    alpha_sum = result["rendered_alpha"].sum(dim=(-1, -2, -3))
    contribution_error = (
        (contribution_sum - alpha_sum).abs() / alpha_sum.clamp_min(1e-8)
    )
    feature_attribution_error = relative_attribution_error(
        result["per_view_feature_cue_sum"],
        pixel_feature_error,
        result["rendered_alpha"],
    )
    rgb_attribution_error = relative_attribution_error(
        result["per_view_rgb_cue_sum"],
        pixel_rgb_error,
        result["rendered_alpha"],
    )
    if float(contribution_error.max()) > 1e-5:
        raise RuntimeError("Contribution conservation check failed")
    if float(feature_attribution_error.max()) > 1e-5:
        raise RuntimeError("Feature-error attribution conservation check failed")
    if float(rgb_attribution_error.max()) > 1e-5:
        raise RuntimeError("RGB-error attribution conservation check failed")
    if grads_before != grads_after:
        raise RuntimeError("Probe polluted encoder parameter gradients")

    zero_lifting_abs_max = None
    if not args.skip_zero_lifting:
        zero_result = decoder.lift_pixel_error_in_chunks(
            gaussians,
            extrinsics,
            intrinsics,
            context["near"],
            context["far"],
            (height, width),
            torch.zeros_like(pixel_feature_error),
            denominator=result["total"],
            chunk_size=lifting_cfg.chunk_size,
            max_probe_channels=lifting_cfg.max_probe_channels,
            eps=lifting_cfg.contribution_eps,
        )
        zero_lifting_abs_max = float(zero_result["cue"].abs().max())
        if zero_lifting_abs_max != 0:
            raise RuntimeError("Zero pixel feature error did not lift to exact zero")

    encoder.refiner.train()
    refined_gaussians, diagnostics = encoder.refiner(
        gaussians, result["feature_cue"], result["rgb_cue"]
    )
    identity_difference = max_gaussian_difference(gaussians, refined_gaussians)
    primary_identity_abs_max = max(
        value for name, value in identity_difference.items() if name != "covariances"
    )
    rebuilt_covariance_error = float(
        (
            refined_gaussians.covariances.float()
            - build_covariance(
                gaussians.scales.detach().float(),
                gaussians.rotations.detach().float(),
            )
        ).abs().max()
    )
    if primary_identity_abs_max > 1e-6 or rebuilt_covariance_error > 1e-6:
        raise RuntimeError(
            "Zero-initialized update head changed parent Gaussians: "
            f"{identity_difference}, covariance={rebuilt_covariance_error}"
        )

    refined_render = decoder.forward(
        refined_gaussians,
        extrinsics,
        intrinsics,
        context["near"],
        context["far"],
        (height, width),
    )
    loss = (refined_render.color.float() - context["image"].float()).square().mean()
    loss.backward()
    refiner_grads = []
    frozen_grads = []
    for name, parameter in encoder.named_parameters():
        if name.startswith("refiner."):
            if parameter.requires_grad and parameter.grad is not None:
                refiner_grads.append(parameter.grad)
        elif parameter.grad is not None:
            frozen_grads.append(name)
    if not refiner_grads or not all(torch.isfinite(grad).all() for grad in refiner_grads):
        raise RuntimeError("Refiner backward produced missing/non-finite gradients")
    if frozen_grads:
        raise RuntimeError(f"Frozen parameters received gradients: {frozen_grads[:10]}")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            refined_gaussians.means,
            refined_gaussians.scales,
            refined_gaussians.rotations,
            refined_gaussians.opacities,
            refined_gaussians.harmonics,
            refined_render.color,
            refined_render.depth,
        )
    ):
        raise RuntimeError("Refined Gaussians/rendering contain NaN or Inf")

    summary = {
        "scene": str(batch["scene"][0]),
        "checkpoint_global_step": checkpoint_step,
        "views": views,
        "gaussians": gaussian_count,
        "pixel_feature_error_shape": list(pixel_feature_error.shape),
        "pixel_rgb_error_shape": list(pixel_rgb_error.shape),
        "gaussian_feature_error_shape": list(result["feature_cue"].shape),
        "gaussian_rgb_error_shape": list(result["rgb_cue"].shape),
        "fused_error_shape": list(fused_cue.shape),
        "pt_input_channels": encoder.refiner.gaussian_dim + 256,
        "feature_probe_count": int(result["feature_probe_count"]),
        "feature_lifting_seconds": float(result["feature_lifting_seconds"]),
        "mean_feature_probe_seconds": float(result["feature_chunk_seconds"]),
        "contribution_conservation_relative_error_max": float(
            contribution_error.max()
        ),
        "feature_attribution_relative_error_max": float(
            feature_attribution_error.max()
        ),
        "rgb_attribution_relative_error_max": float(rgb_attribution_error.max()),
        "zero_lifting_abs_max": zero_lifting_abs_max,
        "encoder_parameters_with_grad_before_probe": grads_before,
        "encoder_parameters_with_grad_after_probe": grads_after,
        "refiner_parameters_with_grad": len(refiner_grads),
        "frozen_parameters_with_grad": len(frozen_grads),
        "zero_initialized_identity_abs_max": identity_difference,
        "zero_initialized_rebuilt_covariance_abs_max": rebuilt_covariance_error,
        "initial_context_psnr": float(psnr(context["image"], initial_render)),
        "refined_context_psnr": float(
            psnr(context["image"], refined_render.color.detach())
        ),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    summary["context_psnr_delta"] = (
        summary["refined_context_psnr"] - summary["initial_context_psnr"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_cue": result["feature_cue"].cpu(),
            "rgb_cue": result["rgb_cue"].cpu(),
            "fused_cue": fused_cue.cpu(),
            "contribution": result["total"].cpu(),
            "radii": result["radii"].cpu(),
        },
        args.output_dir / "gaussian_256d_error_cue.pt",
    )
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
