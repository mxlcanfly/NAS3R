from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any, Dict

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F
import json
import numpy as np
import cv2
import os
import time

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections, render_cameras_es
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.frozen_nas3rm_depth_teacher import FrozenNAS3RMDepthTeacher
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..misc.intrinsics_utils import estimate_intrinsics
from ..evaluation.metrics import compute_pose_error, compute_pose_error_for_batch
from ..misc.cam_utils import pose_auc
from .ply_export import export_parent_child_debug_gaussians, export_ply


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float
    min_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    child_hr_evaluation: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    training_context: bool
    freeze_pretrained: bool
    freeze_backbone: bool
    freeze_pose_head: bool
    freeze_aggregator: bool
    freeze_intrinsics_head: bool

    random_drop_context_views: bool = False
    random_drop_target_views: bool = False

    pretrain_camera_head: bool = False
    child_gaussian_lowfreq_supervision: bool = False
    child_lowfreq_visualization_interval: int = 0
    child_depth_consistency_weight: float = 0.0
    hr_depth_teacher_weight_path: str | None = None
    hr_depth_supervision_weight: float = 0.0


def dropout_context_views(v_cxt):
    assert v_cxt >= 2, "Need at least 2 context views to preserve left and right"

    # Always keep the first and last view
    left_idx = 0
    right_idx = v_cxt - 1
    if v_cxt > 2:
        # Indices between left and right (exclusive)
        middle_indices = torch.arange(1, v_cxt - 1)
        # Randomly decide how many to keep (can be zero)
        num_keep = torch.randint(low=0, high=len(middle_indices) + 1, size=()).item()
        kept_middle = middle_indices[torch.randperm(len(middle_indices))[:num_keep]]
        selected_indices = torch.cat([torch.tensor([left_idx]), kept_middle, torch.tensor([right_idx])])
        selected_indices, _ = torch.sort(selected_indices)
    else:
        selected_indices = torch.tensor([left_idx, right_idx])  # Only 2 views
    return selected_indices


def dropout_target_views(v, num_keep=None):
    # keep at lease one view for gaussians
    all_indices = torch.arange(0, v)
    if num_keep is None:
        num_keep = torch.randint(low=1, high=v + 1, size=()).item()
    kept_indices = all_indices[torch.randperm(len(all_indices))[:num_keep]]
    kept_indices, _ = torch.sort(kept_indices)
    return kept_indices


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
            self,
            t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
            self,
            optimizer_cfg: OptimizerCfg,
            test_cfg: TestCfg,
            train_cfg: TrainCfg,
            encoder: Encoder,
            encoder_visualizer: Optional[EncoderVisualizer],
            decoder: Decoder,
            losses: list[Loss],
            step_tracker: StepTracker | None,
            distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.hr_depth_teacher = None
        if (
            self.train_cfg.hr_depth_supervision_weight > 0
            and self.train_cfg.hr_depth_teacher_weight_path
        ):
            if getattr(self.encoder.cfg, "name", None) != "nas3r-m":
                raise ValueError("HR depth teacher supervision currently requires the nas3r-m encoder")
            self.hr_depth_teacher = FrozenNAS3RMDepthTeacher(
                self.encoder.cfg,
                self.train_cfg.hr_depth_teacher_weight_path,
            )
            self.hr_depth_teacher.requires_grad_(False)
            self.hr_depth_teacher.eval()

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.test_step_outputs = {}

        # Metric tracking state (used in print_preview_metrics and on_test_end)
        self.running_metrics = None
        self.running_metric_steps = 0
        self.all_metrics = {}
        self.running_metrics_sub = None
        self.running_metric_steps_sub = {}
        self.all_metrics_sub = {}

        self.ckpt_path = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep the externally loaded frozen depth teacher out of train checkpoints."""
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        prefix = "hr_depth_teacher."
        for key in [key for key in state_dict if key.startswith(prefix)]:
            del state_dict[key]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore strict loading with the teacher initialized from its external checkpoint."""
        if self.hr_depth_teacher is None:
            return
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        prefix = "hr_depth_teacher."
        for key, value in self.hr_depth_teacher.state_dict().items():
            state_dict.setdefault(f"{prefix}{key}", value)

    def _image_key(self, views: dict) -> str:
        if getattr(self.encoder.cfg, "name", None) == "nas3r-m" and "image_lr" in views:
            return "image_lr"
        return "image"

    def _images(self, views: dict) -> Tensor:
        return views[self._image_key(views)]

    def _checkpoint_interval(self) -> int | None:
        checkpoint_callbacks = getattr(self.trainer, "checkpoint_callbacks", [])
        for callback in checkpoint_callbacks:
            every_n_train_steps = getattr(callback, "every_n_train_steps", None)
            if every_n_train_steps is not None and every_n_train_steps > 0:
                return int(every_n_train_steps)
        callback = getattr(self.trainer, "checkpoint_callback", None)
        every_n_train_steps = getattr(callback, "every_n_train_steps", None)
        if every_n_train_steps is not None and every_n_train_steps > 0:
            return int(every_n_train_steps)
        return None

    def _checkpoint_dir(self) -> Path:
        checkpoint_callbacks = getattr(self.trainer, "checkpoint_callbacks", [])
        for callback in checkpoint_callbacks:
            dirpath = getattr(callback, "dirpath", None)
            if dirpath is not None:
                return Path(dirpath)
        callback = getattr(self.trainer, "checkpoint_callback", None)
        dirpath = getattr(callback, "dirpath", None)
        if dirpath is not None:
            return Path(dirpath)
        return Path("checkpoints")

    def _should_save_child_lowfreq_visualization(self) -> bool:
        if self.global_rank != 0 or not self.train_cfg.child_gaussian_lowfreq_supervision:
            return False
        interval = self.train_cfg.child_lowfreq_visualization_interval or self._checkpoint_interval()
        step_after_batch = self.global_step + 1
        return interval is not None and step_after_batch > 0 and step_after_batch % interval == 0

    @staticmethod
    def _resize_image_for_vis(image: Tensor, size: tuple[int, int]) -> Tensor:
        if image.shape[-2:] == size:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @staticmethod
    def _depth_for_vis(depth: Tensor) -> Tensor:
        valid = torch.isfinite(depth) & (depth > 0)
        if not valid.any():
            return torch.zeros(3, *depth.shape[-2:], dtype=depth.dtype, device=depth.device)
        log_depth = depth.clamp_min(1e-6).log()
        near = log_depth[valid].quantile(0.01)
        far = log_depth[valid].quantile(0.99)
        denom = (far - near).clamp_min(1e-6)
        normalized = 1.0 - (log_depth - near) / denom
        return apply_color_map_to_image(normalized.clamp(0.0, 1.0), "turbo")

    @staticmethod
    def _depths_for_shared_vis(*depths: Tensor) -> list[Tensor]:
        """Colorize depth maps with one shared robust log-depth range."""
        valid_values = [
            depth[torch.isfinite(depth) & (depth > 0)].detach()
            for depth in depths
        ]
        valid_values = [values for values in valid_values if values.numel() > 0]
        if not valid_values:
            return [
                torch.zeros(3, *depth.shape[-2:], dtype=depth.dtype, device=depth.device)
                for depth in depths
            ]
        log_values = torch.cat(valid_values).clamp_min(1e-6).log()
        near = log_values.quantile(0.01)
        far = log_values.quantile(0.99)
        denom = (far - near).clamp_min(1e-6)
        visualizations = []
        for depth in depths:
            normalized = 1.0 - (depth.clamp_min(1e-6).log() - near) / denom
            valid = torch.isfinite(depth) & (depth > 0)
            normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
            visualizations.append(apply_color_map_to_image(normalized.clamp(0.0, 1.0), "turbo"))
        return visualizations

    @staticmethod
    def _error_for_vis(error: Tensor) -> Tensor:
        finite = error[torch.isfinite(error)].detach()
        if finite.numel() == 0:
            normalized = torch.zeros_like(error)
        else:
            scale = finite.quantile(0.99).clamp_min(1e-6)
            normalized = torch.nan_to_num(error / scale, nan=0.0, posinf=1.0, neginf=0.0)
        return apply_color_map_to_image(normalized.clamp(0.0, 1.0), "inferno")

    @staticmethod
    def _psnr_for_vis(pred: Tensor, target: Tensor) -> float:
        mse = (pred.float() - target.float()).square().mean().clamp_min(1e-12)
        return float((-10.0 * torch.log10(mse)).detach().cpu())

    def _save_child_lowfreq_visualization(
        self,
        batch: BatchedExample,
        output,
        context_output,
        parent_gaussians,
        child_gaussians,
        child_decode: dict[str, Tensor] | None = None,
        teacher_hr_depth: Tensor | None = None,
        ) -> None:
        step_dir = self._checkpoint_dir() / "child_lowfreq_vis" / f"{self.global_step + 1:0>6}"
        image_path = step_dir / "hr_teacher_child_depth_comparison.png"
        ply_path = step_dir / "child_gaussians_supersplat.ply"
        group_supersplat_path = step_dir / "parent_child_groups_supersplat.ply"
        group_stats_path = step_dir / "parent_child_stats.npz"
        group_stats_csv_path = step_dir / "parent_child_stats.csv"

        target_hr = batch["target"]["image"][0, 0].detach()
        context_hr = batch["context"]["image"][0, 0].detach().clamp(0.0, 1.0)
        render_target = output.color[0, 0].detach().clamp(0.0, 1.0)
        render_context = context_output.color[0, 0].detach().clamp(0.0, 1.0)
        scene_psnr = self._psnr_for_vis(render_target, target_hr.clamp(0.0, 1.0))

        if teacher_hr_depth is None:
            teacher_depth = context_output.depth[0, 0].detach()
            teacher_label = "Teacher HR Depth N/A"
        else:
            teacher_depth = teacher_hr_depth[0, 0].detach()
            teacher_label = "Frozen NAS3R Teacher HR Depth"
        child_raw_depth = context_output.depth[0, 0].detach()
        if context_output.alpha is None:
            child_alpha = torch.ones_like(child_raw_depth)
        else:
            child_alpha = context_output.alpha[0, 0].detach().clamp(0.0, 1.0)
        child_expected_depth = torch.where(
            child_alpha > 1e-3,
            child_raw_depth / child_alpha.clamp_min(1e-6),
            torch.full_like(child_raw_depth, float("nan")),
        )
        teacher_depth_vis, child_raw_depth_vis, child_expected_depth_vis = self._depths_for_shared_vis(
            teacher_depth,
            child_raw_depth,
            child_expected_depth,
        )
        depth_error_vis = self._error_for_vis((child_raw_depth - teacher_depth).abs())
        alpha_vis = child_alpha.unsqueeze(0).repeat(3, 1, 1)

        rgb_row = hcat(
            add_label(context_hr, "GT Context HR"),
            add_label(render_context, "Child GS Context Render"),
            add_label((render_context - context_hr).abs().clamp(0.0, 1.0), "Context RGB Abs Error"),
            add_label(target_hr.clamp(0.0, 1.0), "GT Target HR"),
            add_label(render_target, f"Child GS Target PSNR {scene_psnr:.2f}"),
        )
        depth_row = hcat(
            add_label(teacher_depth_vis, teacher_label),
            add_label(child_raw_depth_vis, "Child GS HR Raw Depth"),
            add_label(child_expected_depth_vis, "Child GS HR Depth / Alpha"),
            add_label(depth_error_vis, "Raw Depth Abs Error"),
            add_label(alpha_vis, "Child GS HR Alpha"),
        )
        comparison = vcat(rgb_row, depth_row)
        save_image(add_border(comparison), image_path)

        export_ply(
            batch["target"]["extrinsics"][0, 0].detach(),
            child_gaussians.means[0].detach(),
            child_gaussians.scales[0].detach(),
            child_gaussians.rotations[0].detach(),
            child_gaussians.harmonics[0].detach(),
            child_gaussians.opacities[0].detach(),
            ply_path,
            save_sh_dc_only=True,
            view_transform=True,
        )
        self._save_parent_child_debug(
            batch["target"]["extrinsics"][0, 0].detach(),
            parent_gaussians,
            child_gaussians,
            child_decode,
            group_supersplat_path,
            group_stats_path,
            group_stats_csv_path,
        )

    def _log_child_gaussian_stats(
        self,
        child_gaussians,
        child_decode: dict[str, Tensor] | None,
        batch_size: int,
    ) -> None:
        scales = child_gaussians.scales.detach()
        opacities = child_gaussians.opacities.detach()
        stats = {
            "child_gs/scale_mean": scales.mean(),
            "child_gs/scale_std": scales.std(unbiased=False),
            "child_gs/scale_min": scales.amin(),
            "child_gs/scale_max": scales.amax(),
            "child_gs/opacity_mean": opacities.mean(),
            "child_gs/opacity_std": opacities.std(unbiased=False),
            "child_gs/opacity_min": opacities.amin(),
            "child_gs/opacity_max": opacities.amax(),
            "child_gs/opacity_below_0.05": (opacities < 0.05).float().mean(),
            "child_gs/opacity_above_0.95": (opacities > 0.95).float().mean(),
        }
        if child_decode is not None:
            local_offset_uv = child_decode.get("local_offset_uv")
            if local_offset_uv is not None:
                local_offset_uv = local_offset_uv.detach()
                stats.update({
                    "child_gs/offset_u_mean": local_offset_uv[..., 0].mean(),
                    "child_gs/offset_u_min": local_offset_uv[..., 0].amin(),
                    "child_gs/offset_u_max": local_offset_uv[..., 0].amax(),
                    "child_gs/offset_v_mean": local_offset_uv[..., 1].mean(),
                    "child_gs/offset_v_min": local_offset_uv[..., 1].amin(),
                    "child_gs/offset_v_max": local_offset_uv[..., 1].amax(),
                })
            delta_z = child_decode.get("predicted_delta_z")
            if delta_z is not None:
                delta_z = delta_z.detach()
                stats.update({
                    "child_gs/delta_z_mean": delta_z.mean(),
                    "child_gs/delta_z_std": delta_z.std(unbiased=False),
                    "child_gs/delta_z_abs_mean": delta_z.abs().mean(),
                    "child_gs/delta_z_abs_max": delta_z.abs().amax(),
                })
            child_depths = child_decode.get("child_depths")
            if child_depths is not None:
                child_depths = child_depths.detach()
                stats.update({
                    "child_gs/depth_mean": child_depths.mean(),
                    "child_gs/depth_min": child_depths.amin(),
                    "child_gs/depth_max": child_depths.amax(),
                    "child_gs/depth_nonpositive_ratio": (child_depths <= 0).float().mean(),
                })
            delta_means = child_decode.get("delta_means")
            if delta_means is not None:
                offset_norm = delta_means.detach().norm(dim=-1)
                stats.update({
                    "child_gs/world_offset_mean": offset_norm.mean(),
                    "child_gs/world_offset_max": offset_norm.amax(),
                })
        self.log_dict(
            stats,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _log_new_module_gradient_stats(self) -> None:
        module_names = getattr(self.encoder.cfg, "trainable_new_modules", [])
        stats = {}
        for module_name in module_names:
            module = getattr(self.encoder, module_name, None)
            if module is None:
                continue
            parameters = [param for param in module.parameters() if param.requires_grad]
            if not parameters:
                continue
            total_numel = sum(param.numel() for param in parameters)
            parameters_with_grad = [param for param in parameters if param.grad is not None]
            grad_numel = sum(param.numel() for param in parameters_with_grad)
            device = parameters[0].device
            if parameters_with_grad:
                grad_sq_sum = torch.stack([
                    param.grad.detach().float().square().sum()
                    for param in parameters_with_grad
                ]).sum()
                grad_abs_max = torch.stack([
                    param.grad.detach().float().abs().amax()
                    for param in parameters_with_grad
                ]).amax()
                grad_norm = grad_sq_sum.sqrt()
            else:
                grad_norm = torch.zeros((), device=device)
                grad_abs_max = torch.zeros((), device=device)
            stats.update({
                f"trainable_modules/{module_name}_grad_coverage": torch.tensor(
                    grad_numel / max(total_numel, 1),
                    device=device,
                ),
                f"trainable_modules/{module_name}_grad_norm": grad_norm,
                f"trainable_modules/{module_name}_grad_abs_max": grad_abs_max,
            })
        if stats:
            self.log_dict(
                stats,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
            )

    @staticmethod
    def _save_parent_child_debug(
        extrinsics: Tensor,
        parent_gaussians,
        child_gaussians,
        child_decode: dict[str, Tensor] | None,
        supersplat_path: Path,
        stats_path: Path,
        stats_csv_path: Path,
        max_groups: int = 12,
    ) -> None:
        if child_decode is None or "delta_means" not in child_decode:
            return
        parent_means = parent_gaussians.means[0].detach()
        child_means = child_gaussians.means[0].detach()
        num_parents = parent_means.shape[0]
        if num_parents == 0 or child_means.shape[0] % num_parents != 0:
            return

        num_children = child_means.shape[0] // num_parents
        child_scales = child_gaussians.scales[0].detach().view(num_parents, num_children, 3)
        child_opacities = child_gaussians.opacities[0].detach().view(num_parents, num_children)
        delta_means = child_decode["delta_means"][0].detach().view(num_parents, num_children, 3)
        offset_norm = delta_means.norm(dim=-1)

        knn_scale = child_decode.get("knn_scale")
        if knn_scale is not None:
            knn_scale = knn_scale[0].detach().view(num_parents).clamp_min(1e-8)
            normalized_offset = offset_norm.max(dim=1).values / knn_scale
        else:
            knn_scale = torch.full((num_parents,), float("nan"), device=parent_means.device)
            normalized_offset = offset_norm.max(dim=1).values

        scale_score = child_scales.amax(dim=(1, 2))
        opacity_score = child_opacities.amax(dim=1)
        score = normalized_offset + scale_score / scale_score.median().clamp_min(1e-8) + opacity_score
        count = min(max_groups, num_parents)
        selected = torch.topk(score, k=count, largest=True).indices

        export_parent_child_debug_gaussians(
            extrinsics,
            parent_means,
            child_means,
            selected,
            num_children,
            supersplat_path,
            as_text=False,
        )

        selected_cpu = selected.detach().cpu().numpy()
        selected_offset_norm = offset_norm[selected].detach().cpu().float().numpy()
        selected_knn_scale = knn_scale[selected].detach().cpu().float().numpy()
        selected_child_scales = child_scales[selected].detach().cpu().float().numpy()
        selected_child_opacities = child_opacities[selected].detach().cpu().float().numpy()
        selected_score = score[selected].detach().cpu().float().numpy()
        stats_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(
            stats_path,
            parent_indices=selected_cpu,
            num_children=np.array(num_children, dtype=np.int64),
            parent_means=parent_means[selected].detach().cpu().float().numpy(),
            child_means=child_means.view(num_parents, num_children, 3)[selected].detach().cpu().float().numpy(),
            delta_means=delta_means[selected].detach().cpu().float().numpy(),
            offset_norm=selected_offset_norm,
            knn_scale=selected_knn_scale,
            child_scales=selected_child_scales,
            child_opacities=selected_child_opacities,
            score=selected_score,
        )
        rows = []
        for row_i, parent_idx in enumerate(selected_cpu):
            rows.append(
                [
                    int(parent_idx),
                    float(selected_score[row_i]),
                    float(selected_knn_scale[row_i]),
                    float(selected_offset_norm[row_i].max()),
                    float(selected_offset_norm[row_i].mean()),
                    float(selected_child_scales[row_i].max()),
                    float(selected_child_scales[row_i].mean()),
                    float(selected_child_opacities[row_i].max()),
                    float(selected_child_opacities[row_i].mean()),
                ]
            )
        np.savetxt(
            stats_csv_path,
            np.asarray(rows, dtype=np.float32),
            delimiter=",",
            header="parent_index,score,knn_scale,max_offset,mean_offset,max_child_scale,mean_child_scale,max_child_opacity,mean_child_opacity",
            comments="",
        )

    @staticmethod
    def _depth_consistency_loss(
        render_depth: Tensor,
        target_depth: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compare rendered context depth downsampled to LR with encoder LR DPT depth.

        render_depth: [B, V, H, W]
        target_depth: [B, V, h, w]
        """
        if render_depth.shape[:2] != target_depth.shape[:2]:
            raise ValueError(
                f"Depth view shape mismatch: render {tuple(render_depth.shape)}, target {tuple(target_depth.shape)}"
            )
        render_flat = rearrange(render_depth, "b v h w -> (b v) () h w")
        target_flat = rearrange(target_depth.detach(), "b v h w -> (b v) () h w")
        if render_flat.shape[-2:] != target_flat.shape[-2:]:
            render_flat = F.interpolate(render_flat, size=target_flat.shape[-2:], mode="bilinear", align_corners=False)
        valid = (
            torch.isfinite(render_flat)
            & torch.isfinite(target_flat)
            & (render_flat > 0)
            & (target_flat > 0)
        )
        if not valid.any():
            zero = render_depth.new_zeros(())
            return zero, {
                "valid_ratio": zero,
                "render_mean": zero,
                "target_mean": zero,
                "abs_error_mean": zero,
            }

        render_valid = render_flat[valid]
        target_valid = target_flat[valid]
        loss = F.l1_loss(render_valid, target_valid)
        stats = {
            "valid_ratio": valid.float().mean(),
            "render_mean": render_valid.mean(),
            "target_mean": target_valid.mean(),
            "abs_error_mean": (render_valid - target_valid).abs().mean(),
        }
        return loss, stats

    @staticmethod
    def _unmasked_pearson_depth_loss(pred_depth: Tensor, target_depth: Tensor) -> Tensor:
        """Compute per-view PCC loss without confidence or validity masking."""
        pred = rearrange(pred_depth, "b v h w -> (b v) (h w)")
        target = rearrange(target_depth.detach(), "b v h w -> (b v) (h w)")
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        pred = pred / (pred.std(dim=-1, keepdim=True, unbiased=False) + 1e-6)
        target = target / (target.std(dim=-1, keepdim=True, unbiased=False) + 1e-6)
        correlation = (pred * target).mean(dim=-1)
        return (1.0 - correlation).mean()

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        if self.train_cfg.random_drop_context_views:
            v_cxt = batch["context"]["image"].shape[1]
            selected_indices = dropout_context_views(v_cxt)
            for key in ["image", "image_lr", "intrinsics", "extrinsics", "index", "near", "far"]:
                if key in batch["context"]:
                    batch["context"][key] = batch["context"][key][:, selected_indices]

        if self.train_cfg.random_drop_target_views:
            v_tgt = batch["target"]["image"].shape[1]
            selected_indices = dropout_target_views(v_tgt)
            for key in batch["target"].keys():
                batch["target"][key] = batch["target"][key][:, selected_indices]

        target_gt = batch["target"]["image"] if not self.train_cfg.training_context else torch.cat(
            [batch["context"]["image"], batch["target"]["image"]], dim=1)
        b, v_tgt, _, h, w = batch["target"]["image"].shape
        v_cxt = batch["context"]["image"].shape[1]

        # Run the model.
        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump,
                                      target=batch["target"] if self.encoder.cfg.estimating_pose else None)

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics, pred_extrinsics_cwt = encoder_output['extrinsics']['c'], encoder_output['extrinsics'][
                'cwt']
            target_extrinsics = pred_extrinsics_cwt[:, v_cxt:]
            context_extrinsics = pred_extrinsics_cwt[:, :v_cxt]
        else:
            target_extrinsics = batch["target"]["extrinsics"]
            context_extrinsics = batch["context"]["extrinsics"]

        if self.encoder.cfg.estimating_focal:
            pred_intrinsics_cwt = encoder_output['intrinsics']['cwt']
            target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
            context_intrinsics = pred_intrinsics_cwt[:, :v_cxt]
            # print("estimate focal fx", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,0,0,0]*w)
        else:
            target_intrinsics = batch["target"]["intrinsics"]
            context_intrinsics = batch["context"]["intrinsics"]

        total_loss = 0
        loss_terms: dict[str, Tensor] = {}

        gaussians = encoder_output["gaussians"]
        use_child_lowfreq = (
            self.train_cfg.child_gaussian_lowfreq_supervision
            and "lr_child_gaussians" in encoder_output
        )
        decoder_gaussians = encoder_output["lr_child_gaussians"] if use_child_lowfreq else gaussians
        # Determine decoder inputs
        extrinsics = target_extrinsics if not self.train_cfg.training_context else torch.cat(
            [context_extrinsics, target_extrinsics], dim=1)
        intrinsics = target_intrinsics if not self.train_cfg.training_context else torch.cat(
            [context_intrinsics, target_intrinsics], dim=1)
        near = batch["target"]["near"] if not self.train_cfg.training_context else torch.cat(
            [batch["context"]["near"], batch["target"]["near"]], dim=1)
        far = batch["target"]["far"] if not self.train_cfg.training_context else torch.cat(
            [batch["context"]["far"], batch["target"]["far"]], dim=1)
        # Run decoder
        output = self.decoder.forward(
            decoder_gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        loss_color = output.color
        loss_target = target_gt
        if use_child_lowfreq:
            self.log("train/child_hr_supervision", 1.0)
            self._log_child_gaussian_stats(
                decoder_gaussians,
                encoder_output.get("lr_child_decode"),
                batch_size=b,
            )

        # Compute PSNR
        psnr = compute_psnr(
            rearrange(loss_target, "b v c h w -> (b v) c h w"),
            rearrange(loss_color, "b v c h w -> (b v) c h w"),
        )
        self.log(f"train/psnr", psnr.mean())

        # Compute and log loss.
        for loss_fn in self.losses:
            if loss_fn.name in ['mse', 'lpips']:
                loss = loss_fn.forward(loss_color, loss_target, decoder_gaussians, self.global_step)
                self.log(
                    f"loss/{loss_fn.name}",
                    loss,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=b,
                )
                loss_terms[loss_fn.name] = loss.detach()
                total_loss += loss

        depth_consistency_weight = self.train_cfg.child_depth_consistency_weight
        context_output_for_depth = None
        lr_depth_for_depth = None
        if (
            use_child_lowfreq
            and depth_consistency_weight > 0
        ):
            context_output_for_depth = self.decoder.forward(
                decoder_gaussians,
                context_extrinsics,
                context_intrinsics,
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            lr_depth_for_depth = encoder_output["lr_dpt_depth"]
            depth_consistency, depth_stats = self._depth_consistency_loss(
                context_output_for_depth.depth,
                lr_depth_for_depth,
            )
            weighted_depth_consistency = depth_consistency_weight * depth_consistency
            self.log_dict(
                {
                    "loss/child_depth_consistency_raw": depth_consistency,
                    "loss/child_depth_consistency_weighted": weighted_depth_consistency,
                    "depth/lr_valid_ratio": depth_stats["valid_ratio"],
                    "depth/render_lr_mean": depth_stats["render_mean"],
                    "depth/lr_dpt_mean": depth_stats["target_mean"],
                    "depth/abs_error_mean": depth_stats["abs_error_mean"],
                },
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
                batch_size=b,
            )
            loss_terms["depth"] = weighted_depth_consistency.detach()
            total_loss += weighted_depth_consistency

        hr_depth_weight = self.train_cfg.hr_depth_supervision_weight
        teacher_hr_depth = None
        if (
            use_child_lowfreq
            and hr_depth_weight > 0
            and self.hr_depth_teacher is not None
        ):
            if context_output_for_depth is None:
                context_output_for_depth = self.decoder.forward(
                    decoder_gaussians,
                    context_extrinsics,
                    context_intrinsics,
                    batch["context"]["near"],
                    batch["context"]["far"],
                    batch["context"]["image"].shape[-2:],
                    depth_mode=self.train_cfg.depth_mode,
                )
            teacher_hr_depth = self.hr_depth_teacher(
                batch["context"]["image"],
                batch["context"]["intrinsics"],
            ).detach()
            rendered_hr_depth = context_output_for_depth.depth
            if rendered_hr_depth.shape != teacher_hr_depth.shape:
                raise ValueError(
                    "HR depth shape mismatch: "
                    f"rendered {tuple(rendered_hr_depth.shape)} vs "
                    f"teacher {tuple(teacher_hr_depth.shape)}"
                )

            hr_depth_l1 = F.l1_loss(rendered_hr_depth, teacher_hr_depth)
            hr_depth_pcc = self._unmasked_pearson_depth_loss(
                rendered_hr_depth,
                teacher_hr_depth,
            )
            hr_depth_raw = hr_depth_l1 + hr_depth_pcc
            weighted_hr_depth = hr_depth_weight * hr_depth_raw
            if context_output_for_depth.alpha is None:
                render_alpha = torch.ones_like(rendered_hr_depth)
            else:
                render_alpha = context_output_for_depth.alpha
            alpha_normalized_depth = torch.where(
                render_alpha > 1e-3,
                rendered_hr_depth / render_alpha.clamp_min(1e-6),
                torch.full_like(rendered_hr_depth, float("nan")),
            )
            valid_alpha_depth = torch.isfinite(alpha_normalized_depth)
            alpha_normalized_mean = (
                alpha_normalized_depth[valid_alpha_depth].mean()
                if valid_alpha_depth.any()
                else rendered_hr_depth.new_zeros(())
            )
            self.log_dict(
                {
                    "loss/hr_depth_l1": hr_depth_l1,
                    "loss/hr_depth_pcc": hr_depth_pcc,
                    "loss/hr_depth_raw": hr_depth_raw,
                    "loss/hr_depth_weighted": weighted_hr_depth,
                    "depth/hr_teacher_mean": teacher_hr_depth.mean(),
                    "depth/hr_render_raw_mean": rendered_hr_depth.mean(),
                    "depth/hr_render_alpha_normalized_mean": alpha_normalized_mean,
                    "depth/hr_render_alpha_mean": render_alpha.mean(),
                    "depth/hr_render_alpha_below_0.1": (render_alpha < 0.1).float().mean(),
                },
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
                batch_size=b,
            )
            loss_terms["hr_depth"] = weighted_hr_depth.detach()
            total_loss += weighted_hr_depth

        self.log(
            "loss/total",
            total_loss,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            batch_size=b,
        )
        loss_terms["total"] = total_loss.detach()

        if use_child_lowfreq and self._should_save_child_lowfreq_visualization():
            with torch.no_grad():
                context_hr_size = batch["context"]["image"].shape[-2:]
                if (
                    context_output_for_depth is None
                    or context_output_for_depth.depth.shape[-2:] != context_hr_size
                ):
                    context_output_for_depth = self.decoder.forward(
                        decoder_gaussians,
                        context_extrinsics,
                        context_intrinsics,
                        batch["context"]["near"],
                        batch["context"]["far"],
                        context_hr_size,
                        depth_mode=self.train_cfg.depth_mode,
                    )
                if teacher_hr_depth is None and self.hr_depth_teacher is not None:
                    teacher_hr_depth = self.hr_depth_teacher(
                        batch["context"]["image"],
                        batch["context"]["intrinsics"],
                    ).detach()
                self._save_child_lowfreq_visualization(
                    batch,
                    output,
                    context_output_for_depth,
                    gaussians,
                    decoder_gaussians,
                    encoder_output.get("lr_child_decode"),
                    teacher_hr_depth,
                )

        if self.encoder.cfg.estimating_pose:
            context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:, v_cxt - 1],
                                                                                   batch["context"]["extrinsics"][
                                                                                       :, v_cxt - 1])
            target_rot_error, target_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:, v_cxt:],
                                                                                 batch["target"]["extrinsics"])

            self.log(f"train/context_angular_error", context_rot_error)
            self.log(f"train/context_transl_error", context_transl_error)
            self.log(f"train/target_angular_error", target_rot_error)
            self.log(f"train/target_transl_error", target_transl_error)

        if (
                self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            loss_text = "; ".join(
                f"{name} = {value.item():.6f}"
                for name, value in loss_terms.items()
            )
            print(
                f"Epoch {self.current_epoch}; "
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"{loss_text}; "
                f"psnr = {psnr.mean().item():.6f}; "
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        v_cxt = batch["context"]["image"].shape[1]
        # NAS3R-M normally uses image_lr through _images(). Child HR evaluation
        # is opt-in because it is substantially more expensive than the legacy
        # parent-GS 64px evaluation.
        target_image_lr = self._images(batch["target"])
        target_image_hr = batch["target"]["image"]
        b, v_tgt, _, h_lr, w_lr = target_image_lr.shape
        assert b == 1

        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        visualization_dump = {}
        use_child_hr_evaluation = False

        if self.encoder.cfg.estimating_pose:
            # Render Gaussians.
            extrinsics_list = []
            rgb_list = []
            for target_view in range(v_tgt):
                # test one by one
                target_data = {
                    "image": batch["target"]["image"][:, target_view:target_view + 1],
                    "intrinsics": batch["target"]["intrinsics"][:, target_view:target_view + 1],
                    "near": batch["target"]["near"][:, target_view:target_view + 1],
                    "far": batch["target"]["far"][:, target_view:target_view + 1],
                }
                if "image_lr" in batch["target"]:
                    target_data["image_lr"] = batch["target"]["image_lr"][:, target_view:target_view + 1]

                with self.benchmarker.time("encoder"):
                    encoder_output = self.encoder(batch["context"], self.global_step,
                                                  visualization_dump=visualization_dump, target=target_data)

                pred_extrinsics_cwt = encoder_output['extrinsics']['cwt']
                gaussians = encoder_output["gaussians"]
                use_child_hr_evaluation = (
                    self.test_cfg.child_hr_evaluation
                    and
                    self.train_cfg.child_gaussian_lowfreq_supervision
                    and "lr_child_gaussians" in encoder_output
                )
                if use_child_hr_evaluation:
                    gaussians = encoder_output["lr_child_gaussians"]
                    render_shape = target_data["image"].shape[-2:]
                else:
                    render_shape = (h_lr, w_lr)

                if self.encoder.cfg.estimating_focal:
                    pred_intrinsics_cwt = encoder_output['intrinsics']['cwt']
                    target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
                    # print("estimate focal", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,target_view,0,0]*w)
                else:
                    target_intrinsics = target_data["intrinsics"]

                if self.test_cfg.align_pose:
                    output, updated_extrinsics = self.test_step_align(target_data, gaussians, target_intrinsics,
                                                                      initial_extrinsics=pred_extrinsics_cwt[:, v_cxt:])

                else:
                    with self.benchmarker.time("decoder", num_calls=1):
                        output = self.decoder.forward(
                            gaussians,
                            pred_extrinsics_cwt[:, v_cxt:],
                            target_intrinsics,
                            batch["target"]["near"][:, target_view:target_view + 1],
                            batch["target"]["far"][:, target_view:target_view + 1],
                            render_shape,
                        )

                extrinsics_list.append(pred_extrinsics_cwt[:, v_cxt:])
                rgb_list.append(output.color)

            target_extrinsics = torch.cat(extrinsics_list, dim=1)  # (b, v, 4, 4)
            rgb_pred = torch.cat(rgb_list, dim=1)[0]  # (v, 3, h, w)


        else:
            # Render Gaussians.
            with self.benchmarker.time("encoder"):
                encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)

            target_extrinsics = batch["target"]["extrinsics"]

            if self.encoder.cfg.estimating_focal:
                pred_intrinsics_cwt = encoder_output['intrinsics']['cwt']
                target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
                # print("estimate focal", target_intrinsics[0,0,0,0]*w, "gt focal", batch["target"]["intrinsics"][0,0,0,0]*w)
            else:
                target_intrinsics = batch["target"]["intrinsics"]

            gaussians = encoder_output['gaussians']
            use_child_hr_evaluation = (
                self.test_cfg.child_hr_evaluation
                and
                self.train_cfg.child_gaussian_lowfreq_supervision
                and "lr_child_gaussians" in encoder_output
            )
            if use_child_hr_evaluation:
                gaussians = encoder_output["lr_child_gaussians"]
                render_shape = target_image_hr.shape[-2:]
            else:
                render_shape = (h_lr, w_lr)

            # align the target pose
            if self.test_cfg.align_pose:
                output, updated_extrinsics = self.test_step_align(batch["target"], gaussians, target_intrinsics)
            else:
                with self.benchmarker.time("decoder", num_calls=v_tgt):
                    output = self.decoder.forward(
                        gaussians,
                        target_extrinsics,
                        target_intrinsics,
                        batch["target"]["near"],
                        batch["target"]["far"],
                        render_shape,
                    )
            rgb_pred = output.color[0]  # (v, 3, h, w)

        (scene,) = batch["scene"]
        rgb_gt = (target_image_hr if use_child_hr_evaluation else target_image_lr)[0]

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            all_metrics = {}

            all_metrics.update({
                f"lpips": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr": compute_psnr(rgb_gt, rgb_pred).mean(),
            })

            if self.encoder.cfg.estimating_pose:
                target_rot_error, target_transl_error = compute_pose_error_for_batch(target_extrinsics,
                                                                                     batch["target"]["extrinsics"])
                context_rot_error, context_transl_error = compute_pose_error_for_batch(
                    pred_extrinsics_cwt[:, v_cxt - 1], batch["context"]["extrinsics"][:, v_cxt - 1])
                target_pose_error = torch.max(target_rot_error, target_transl_error)
                context_pose_error = torch.max(context_rot_error, context_transl_error)
                all_metrics.update({
                    f"tgt_e_R": target_rot_error,
                    f"tgt_e_t": target_transl_error,
                    f"tgt_e_pose": target_pose_error,
                    f"cxt_e_R": context_rot_error,
                    f"cxt_e_t": context_transl_error,
                    f"cxt_e_pose": context_pose_error,
                })

            if scene not in self.test_step_outputs:
                self.test_step_outputs[scene] = [overlap_tag]
                self.test_step_outputs[scene] += [all_metrics[f'psnr'].item(), all_metrics[f'ssim'].item(),
                                                  all_metrics[f'lpips'].item()]

            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_image:
            for index, pred in zip(batch["target"]["index"][0], rgb_pred):
                save_image(pred, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])

            save_video(
                [a for a in rgb_pred],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = self._images(batch["context"])[0]
            comparison = [
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)")
            ]
            save_image(add_border(hcat(*comparison)), path / f"{scene}.png")

    def test_step_align(self, target, gaussians, intrinsics, initial_extrinsics=None):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        target_image = self._images(target)
        b, v, _, h, w = target_image.shape
        device = target_image.device
        with torch.set_grad_enabled(True):
            if initial_extrinsics is not None:
                extrinsics = nn.Parameter(initial_extrinsics)
            else:
                extrinsics = nn.Parameter(target["extrinsics"])

            opt_params = []
            opt_params.append(
                {
                    "params": [extrinsics],
                    "lr": self.test_cfg.opt_lr,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        intrinsics,
                        target["near"],
                        target["far"],
                        (h, w)
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if loss_fn.name in ["mse", "lpips"]:
                            loss = loss_fn.forward(output.color, target_image, gaussians, self.global_step)
                            total_loss = total_loss + loss

                    total_loss.backward()

                    if (i == 0) or (i % 50 == 0) or (i == self.test_cfg.pose_align_steps - 1):
                        print(i, total_loss.item())

                    pose_optimizer.step()

        return output, extrinsics

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

        if self.ckpt_path is not None:
            with open(self.test_cfg.output_path / name / "test_ckpt_path.txt", "w") as f:
                f.write(f"{self.ckpt_path}\n")

        with (self.test_cfg.output_path / name / f"scores_all.json").open("w") as f:
            json.dump(self.test_step_outputs, f, indent=2)

        def convert_tensors_to_values(metrics_dict):
            return {k: convert_tensors_to_values(v) if isinstance(v, dict) else v.item() for k, v in
                    metrics_dict.items()}

        running_metrics = convert_tensors_to_values(self.running_metrics)
        with (self.test_cfg.output_path / name / f"scores_all_avg.json").open("w") as f:
            json.dump(running_metrics, f, indent=2)

        running_metrics_sub = convert_tensors_to_values(self.running_metrics_sub)
        with (self.test_cfg.output_path / name / f"scores_sub_avg.json").open("w") as f:
            json.dump(running_metrics_sub, f, indent=2)

        for item in ['R', 't', 'pose']:
            if self.encoder.cfg.estimating_pose:
                print("*" * 20)
                for method in ['ours']:
                    tot_e_pose = np.array(self.all_metrics[f"cxt_e_{item}"])
                    tot_e_pose = np.array(tot_e_pose)
                    thresholds = [5, 10, 20]
                    auc = pose_auc(tot_e_pose, thresholds)
                    self.running_metrics[f'{item}_auc'] = auc

                    median_error = np.median(tot_e_pose)
                    self.running_metrics[f'{item}_median'] = median_error
                    print(f"Pose AUC {method} of {item}: ", auc, " median error", median_error)

                    for overlap_tag in self.all_metrics_sub.keys():
                        tot_e_pose = np.array(self.all_metrics_sub[overlap_tag][f"cxt_e_{item}"])
                        tot_e_pose = np.array(tot_e_pose)
                        thresholds = [5, 10, 20]
                        auc = pose_auc(tot_e_pose, thresholds)
                        self.running_metrics_sub[overlap_tag][f'{item}_auc'] = auc

                        median_error = np.median(tot_e_pose)
                        self.running_metrics_sub[overlap_tag][f'{item}_median'] = median_error
                        print(f"Pose AUC {method} {overlap_tag} of {item}: ", auc, " median error", median_error)

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Render Gaussians.
        if self.train_cfg.random_drop_context_views:
            v_cxt = batch["context"]["image"].shape[1]
            selected_indices = dropout_context_views(v_cxt)
            # Apply selection to all context elements
            for key in ["image", "image_lr", "intrinsics", "extrinsics", "index", "near", "far"]:
                if key in batch["context"]:
                    batch["context"][key] = batch["context"][key][:, selected_indices]

        if self.train_cfg.random_drop_target_views:
            v_tgt = batch["target"]["image"].shape[1]
            selected_indices = dropout_target_views(v_tgt)
            # Apply selection to all target elements
            for key in batch["target"].keys():
                batch["target"][key] = batch["target"][key][:, selected_indices]

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}"
            )

        v_cxt = batch["context"]["image"].shape[1]
        target_image = self._images(batch["target"])
        context_image = self._images(batch["context"])
        b, v_tgt, _, h, w = target_image.shape
        assert b == 1

        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump,
                                      target=batch["target"] if self.encoder.cfg.estimating_pose else None)

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics, pred_extrinsics_cwt = encoder_output['extrinsics']['c'], encoder_output['extrinsics'][
                'cwt']
            target_extrinsics = pred_extrinsics_cwt[:, v_cxt:]
            context_extrinsics = pred_extrinsics_cwt[:, :v_cxt]
        else:
            target_extrinsics = batch["target"]["extrinsics"]
            context_extrinsics = batch["context"]["extrinsics"]

        if self.encoder.cfg.estimating_focal:
            if "intrinsics" in encoder_output:
                pred_intrinsics_cwt = encoder_output['intrinsics']['cwt']
                target_intrinsics = pred_intrinsics_cwt[:, v_cxt:]
                context_intrinsics = pred_intrinsics_cwt[:, :v_cxt]
            else:
                intrinsics = estimate_intrinsics(visualization_dump['means'].squeeze(-2), h, w)
                target_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_tgt, 1, 1)
                context_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_cxt, 1, 1)
        else:
            target_intrinsics = batch["target"]["intrinsics"]
            context_intrinsics = batch["context"]["intrinsics"]

        gaussians = encoder_output['gaussians']

        # Render context + target views for validation visualization
        extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=1)
        intrinsics = torch.cat([context_intrinsics, target_intrinsics], dim=1)
        near = torch.cat([batch["context"]["near"], batch["target"]["near"]], dim=1)
        far = torch.cat([batch["context"]["far"], batch["target"]["far"]], dim=1)
        target_gt = torch.cat([context_image, target_image], dim=1)

        # Run decoder
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        # Compute validation metrics.
        rgb_gt = target_gt[0]
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # Target view metrics
        psnr = compute_psnr(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
        self.log(f"val/lpips", lpips)
        ssim_val = compute_ssim(rgb_gt[v_cxt:], rgb_pred[v_cxt:]).mean()
        self.log(f"val/ssim", ssim_val)

        # Context view metrics
        psnr = compute_psnr(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
        self.log(f"val/context/psnr", psnr)
        lpips = compute_lpips(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
        self.log(f"val/context/lpips", lpips)
        ssim_val = compute_ssim(rgb_gt[:v_cxt], rgb_pred[:v_cxt]).mean()
        self.log(f"val/context/ssim", ssim_val)

        # Construct comparison image.
        context_img = context_image[0]
        context_img_depth = vis_depth_map(visualization_dump["depth"][0])  # (v, h, w)

        comparison = hcat(
            add_label(vcat(*context_img), "Context"),
            add_label(vcat(*context_img_depth), "Context Depth"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), f"Prediction"),
            add_label(vcat(*depth_pred), f"Depth")
        )

        if self.encoder.cfg.estimating_pose:
            context_rot_error, context_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:, v_cxt - 1],
                                                                                   batch["context"]["extrinsics"][
                                                                                       :, v_cxt - 1])
            target_rot_error, target_transl_error = compute_pose_error_for_batch(pred_extrinsics_cwt[:, v_cxt:],
                                                                                 batch["target"]["extrinsics"])

            self.log(f"val/context_angular_error", context_rot_error)
            self.log(f"val/context_transl_error", context_transl_error)
            self.log(f"val/target_angular_error", target_rot_error)
            self.log(f"val/target_transl_error", target_transl_error)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Run video validation step.
        self.render_video_interpolation(batch)

        if self.train_cfg.extended_visualization:
            self.render_video_wobble(batch)
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t, context_extrinsics, context_intrinsics):
            origin_a = context_extrinsics[:, 0, :3, 3]
            origin_b = context_extrinsics[:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                context_extrinsics[:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                context_intrinsics[:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        def trajectory_fn(t, context_extrinsics, context_intrinsics):
            extrinsics = interpolate_extrinsics(
                context_extrinsics[0, 0],
                context_extrinsics[0, -1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                context_intrinsics[0, 0],
                context_intrinsics[0, -1],
                t,
            )

            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        def trajectory_fn(t, context_extrinsics, context_intrinsics):
            origin_a = context_extrinsics[:, 0, :3, 3]
            origin_b = context_extrinsics[:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                context_extrinsics[0, 0],
                context_extrinsics[0, -1],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                context_intrinsics[0, 0],
                context_intrinsics[0, -1],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
            self,
            batch: BatchedExample,
            trajectory_fn: TrajectoryFn,
            name: str,
            num_frames: int = 30,
            smooth: bool = True,
            loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.

        _, _, _, h, w = self._images(batch["context"]).shape
        _, v_cxt, _, _ = batch["context"]["extrinsics"].shape

        visualization_dump = {}
        encoder_output = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump,
                                      target=None)
        gaussians = encoder_output['gaussians']

        if self.encoder.cfg.estimating_pose:
            pred_extrinsics = encoder_output['extrinsics']['c']
            context_extrinsics = pred_extrinsics[:, :v_cxt]
        else:
            context_extrinsics = batch["context"]["extrinsics"]

        if self.encoder.cfg.estimating_focal:
            pred_intrinsics_cwt = encoder_output['intrinsics']['c']
            context_intrinsics = pred_intrinsics_cwt[:, :v_cxt]
        else:
            context_intrinsics = batch["context"]["intrinsics"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t, context_extrinsics, context_intrinsics)

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )

        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video, fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None,
                              overlap_tag: str | None = None) -> None:
        if self.running_metrics is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
            self.all_metrics = {k: [v.cpu().item()] for k, v in metrics.items()}
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

            for k, v in metrics.items():
                self.all_metrics[k].append(v.cpu().item())

        if overlap_tag is not None:
            if self.running_metrics_sub is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
                self.all_metrics_sub = {overlap_tag: {k: [v.cpu().item()] for k, v in metrics.items()}}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
                self.all_metrics_sub[overlap_tag] = {k: [v.cpu().item()] for k, v in metrics.items()}
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

                for k, v in metrics.items():
                    self.all_metrics_sub[overlap_tag][k].append(v.cpu().item())

        metric_list = list(metrics.keys())

        def print_metrics(running_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{running_metric[f'{metric}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def freeze_params(self, freeze_keywords=None, unfreeze_keywords=None):
        freeze_keywords = freeze_keywords or []
        unfreeze_keywords = unfreeze_keywords or []

        for name, param in self.named_parameters():
            if unfreeze_keywords:
                if any(kw in name for kw in unfreeze_keywords):
                    param.requires_grad = True
                    print(f"Unfreezing: {name}")
                else:
                    param.requires_grad = False
            elif freeze_keywords:
                if any(kw in name for kw in freeze_keywords):
                    param.requires_grad = False
                    print(f"Freezing: {name}")

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []

        train_only_new_nas3rm_modules = (
            getattr(self.encoder.cfg, "name", None) == "nas3r-m"
            and getattr(self.encoder.cfg, "freeze_original_lr_network", False)
        )
        if train_only_new_nas3rm_modules:
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                new_params.append(param)
                new_param_names.append(name)
        else:
            has_pretrained_backbone = getattr(self.encoder.backbone.cfg, "pretrained", False)
            has_pretrained_encoder_weights = bool(getattr(self.encoder.cfg, "pretrained_weights", ""))
            if not (has_pretrained_backbone or has_pretrained_encoder_weights):
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    new_params.append(param)
                    new_param_names.append(name)
            else:
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Heads that are always treated as new
                    if any(x in name for x in ["gaussian_param_head", "intrinsic_encoder"]):
                        new_params.append(param)
                        new_param_names.append(name)
                        # print(name)

                    # Camera head logic
                    elif "camera_head" in name:
                        if self.train_cfg.pretrain_camera_head:
                            pretrained_params.append(param)
                            pretrained_param_names.append(name)
                        else:
                            new_params.append(param)
                            new_param_names.append(name)
                    else:
                        pretrained_params.append(param)
                        pretrained_param_names.append(name)

        optimizer_params = new_params + pretrained_params
        optimizer_param_ids = [id(param) for param in optimizer_params]
        if len(optimizer_param_ids) != len(set(optimizer_param_ids)):
            raise RuntimeError("A trainable parameter was added to the optimizer more than once")
        optimizer_param_id_set = set(optimizer_param_ids)
        missing_trainable = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and id(param) not in optimizer_param_id_set
        ]
        if missing_trainable:
            raise RuntimeError(
                "Trainable parameters missing from optimizer: "
                + ", ".join(missing_trainable[:20])
            )

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
            },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"],
                                                                  eta_min=self.optimizer_cfg.lr * self.optimizer_cfg.min_lr_multiplier)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler],
                                                             milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Perform the backward pass
        optimizer_closure()
        self._log_new_module_gradient_stats()

        nan_detected = False
        large_detected = False

        max_grad_norm = 20 if 'vggt' in self.encoder.backbone.cfg.name else 5  # Define the maximum norm threshold
        max_grad_value = 0

        # Check for NaN gradients
        for name, param in self.named_parameters():

            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    self.log("nan_gradient_detected", True, on_step=True, on_epoch=False)
                    print(f"Skipping update due to NaN gradient")
                    nan_detected = True
                    break

                param_max_grad = param.grad.abs().max().item()
                if param_max_grad > max_grad_value:
                    max_grad_value = param_max_grad

                if param.grad.abs().max() > max_grad_norm:
                    self.log(f"large_gradient_detected in {name}", param.grad.abs().max().item(), on_step=True,
                             on_epoch=False)
                    print(f"large gradient in {name}")
                    large_detected = True
                    break

        self.log("max_gradient", max_grad_value, on_step=True, on_epoch=False)

        if nan_detected or large_detected:
            optimizer.zero_grad()  # Clear gradients if skipping the step
        else:
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
