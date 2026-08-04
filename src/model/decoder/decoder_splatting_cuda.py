from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    enable_cov_grad: bool
    enable_sh_grad: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
            self,
            cfg: DecoderSplattingCUDACfg,
            # dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.enable_cov_grad = cfg.enable_cov_grad
        self.enable_sh_grad = cfg.enable_sh_grad
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
            repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
            scale_invariant=self.make_scale_invariant,
            enable_cov_grad=self.enable_cov_grad,
            enable_sh_grad=self.enable_sh_grad
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        depth = rearrange(depth, "(b v) 1 h w -> b v h w", b=b, v=v)

        if self.make_scale_invariant:
            scale = near / 1
            depth = depth * scale[:, :, None, None]

        return DecoderOutput(color, depth)

    @staticmethod
    def _validate_pixel_cue(
            pixel_cue: Tensor | None,
            batch: int,
            views: int,
            image_shape: tuple[int, int],
            max_channels: int,
    ) -> int:
        if pixel_cue is None:
            return 0
        if pixel_cue.ndim != 5 or pixel_cue.shape[:2] != (batch, views):
            raise ValueError(
                "pixel_cue must have shape [batch, view, channel, height, width], "
                f"got={tuple(pixel_cue.shape)}"
            )
        if tuple(pixel_cue.shape[-2:]) != image_shape:
            raise ValueError(
                f"pixel_cue resolution {tuple(pixel_cue.shape[-2:])} does not "
                f"match image_shape={image_shape}"
            )
        channels = int(pixel_cue.shape[2])
        if channels == 0:
            raise ValueError("pixel_cue must contain at least one channel")
        if channels > max_channels:
            raise ValueError(
                f"pixel_cue has {channels} channels, but this probe supports at "
                f"most {max_channels}"
            )
        return channels

    def _run_gaussian_probe(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            pixel_cue: Tensor | None = None,
            include_contribution: bool = True,
    ) -> dict[str, Tensor]:
        """Run one exact virtual-attribute probe on a fresh rasterization graph."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Gaussian probes cannot run inside "
                "torch.inference_mode(); use torch.no_grad() for the encoder "
                "and call this method outside that context."
            )

        batch, views = extrinsics.shape[:2]
        if gaussians.means.shape[0] != batch:
            raise ValueError(
                "Gaussian and camera batch sizes differ: "
                f"{gaussians.means.shape[0]} != {batch}"
            )
        gaussian_count = gaussians.means.shape[1]
        contribution_channels = int(include_contribution)
        cue_channels = self._validate_pixel_cue(
            pixel_cue,
            batch,
            views,
            image_shape,
            max_channels=34 - contribution_channels,
        )
        probe_channels = contribution_channels + cue_channels
        if probe_channels == 0:
            raise ValueError("A probe needs a pixel cue or a contribution channel")

        # torch.enable_grad() makes this helper usable when its caller is under
        # torch.no_grad(), while autograd.grad avoids populating any parameter
        # .grad fields.
        with torch.enable_grad():
            probe = torch.zeros(
                batch,
                views,
                gaussian_count,
                probe_channels,
                device=gaussians.means.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            _, _, aux = render_cuda(
                rearrange(extrinsics.detach().float(), "b v i j -> (b v) i j"),
                rearrange(intrinsics.detach().float(), "b v i j -> (b v) i j"),
                rearrange(near.detach().float(), "b v -> (b v)"),
                rearrange(far.detach().float(), "b v -> (b v)"),
                image_shape,
                repeat(
                    self.background_color.detach().float(),
                    "c -> (b v) c",
                    b=batch,
                    v=views,
                ),
                repeat(
                    gaussians.means.detach().float(),
                    "b g xyz -> (b v) g xyz",
                    v=views,
                ),
                repeat(
                    gaussians.covariances.detach().float(),
                    "b g i j -> (b v) g i j",
                    v=views,
                ),
                repeat(
                    gaussians.harmonics.detach().float(),
                    "b g c d_sh -> (b v) g c d_sh",
                    v=views,
                ),
                repeat(
                    gaussians.opacities.detach().float(),
                    "b g -> (b v) g",
                    v=views,
                ),
                repeat(
                    gaussians.rotations.detach().float(),
                    "b g i -> (b v) g i",
                    v=views,
                ),
                repeat(
                    gaussians.scales.detach().float(),
                    "b g i -> (b v) g i",
                    v=views,
                ),
                scale_invariant=self.make_scale_invariant,
                enable_cov_grad=self.enable_cov_grad,
                enable_sh_grad=self.enable_sh_grad,
                gaussian_extra_attrs=rearrange(
                    probe,
                    "b v g c -> (b v) g c",
                ),
                return_aux=True,
            )
            rendered_extra = rearrange(
                aux.extra,
                "(b v) c h w -> b v c h w",
                b=batch,
                v=views,
            )
            objective = rendered_extra.new_zeros(())
            cue_start = contribution_channels
            if include_contribution:
                objective = objective + rendered_extra[:, :, 0].sum()
            resolved_pixel_cue = (
                pixel_cue.detach().float() if pixel_cue is not None else None
            )
            if cue_channels:
                objective = objective + (
                        rendered_extra[:, :, cue_start: cue_start + cue_channels]
                        * resolved_pixel_cue
                ).sum()

            probe_gradient = torch.autograd.grad(
                outputs=objective,
                inputs=probe,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        result = {
            "radii": rearrange(
                aux.radii.detach(),
                "(b v) g -> b v g",
                b=batch,
                v=views,
            ),
            "rendered_alpha": rearrange(
                aux.alpha.detach(),
                "(b v) c h w -> b v c h w",
                b=batch,
                v=views,
            ),
        }
        if include_contribution:
            per_view = probe_gradient[..., 0].detach().float()
            result["per_view"] = per_view
            result["total"] = per_view.sum(dim=1)
        if cue_channels:
            per_view_cue_sum = (
                probe_gradient[..., cue_start: cue_start + cue_channels]
                .detach()
                .float()
            )
            cue_sum = per_view_cue_sum.sum(dim=1)
            result.update(
                {
                    "per_view_cue_sum": per_view_cue_sum,
                    "cue_sum": cue_sum,
                    "pixel_cue": resolved_pixel_cue.detach(),
                }
            )
        return result

    @staticmethod
    def _normalize_cue(
            numerator: Tensor,
            denominator: Tensor,
            eps: float,
    ) -> Tensor:
        valid = denominator > eps
        return torch.where(
            valid[..., None],
            numerator / denominator[..., None].clamp_min(eps),
            torch.zeros_like(numerator),
        )

    def compute_gaussian_contribution(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            ground_truth: Float[Tensor, "batch view 3 height width"] | None = None,
            pixel_cue: Tensor | None = None,
            eps: float = 1e-8,
    ) -> dict[str, Tensor]:
        """Return exact contribution and an optional normalized Gaussian cue."""
        if ground_truth is not None and pixel_cue is not None:
            raise ValueError("Pass either ground_truth or pixel_cue, not both")
        if ground_truth is not None:
            expected_shape = (*extrinsics.shape[:2], 3, *image_shape)
            if tuple(ground_truth.shape) != expected_shape:
                raise ValueError(
                    "ground_truth must have shape [batch, view, 3, height, width]: "
                    f"expected={expected_shape}, got={tuple(ground_truth.shape)}"
                )
            with torch.no_grad():
                rendered = self.forward(
                    gaussians,
                    extrinsics,
                    intrinsics,
                    near,
                    far,
                    image_shape,
                ).color
            # Keep the same signed convention as the refiner and ReSplat:
            # prediction minus observation.
            pixel_cue = rendered.detach().float() - ground_truth.detach().float()

        result = self._run_gaussian_probe(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            pixel_cue=pixel_cue,
            include_contribution=True,
        )
        if pixel_cue is not None:
            per_view = result["per_view"]
            total = result["total"]
            result["per_view_cue"] = self._normalize_cue(
                result["per_view_cue_sum"], per_view, eps
            )
            result["cue"] = self._normalize_cue(result["cue_sum"], total, eps)
        return result

    def lift_pixel_error_in_chunks(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            pixel_error: Tensor,
            denominator: Tensor,
            chunk_size: int = 32,
            max_probe_channels: int = 34,
            eps: float = 1e-8,
    ) -> dict[str, Tensor]:
        """Lift arbitrary signed pixel channels using independent probe graphs."""
        if max_probe_channels <= 0 or max_probe_channels > 34:
            raise ValueError(
                "max_probe_channels must respect the CUDA limit [1, 34], got "
                f"{max_probe_channels}"
            )
        if chunk_size <= 0 or chunk_size > max_probe_channels:
            raise ValueError(
                f"chunk_size must be in [1, {max_probe_channels}], got {chunk_size}"
            )
        channels = self._validate_pixel_cue(
            pixel_error,
            extrinsics.shape[0],
            extrinsics.shape[1],
            image_shape,
            max_channels=int(pixel_error.shape[2]),
        )
        if denominator.shape != gaussians.means.shape[:2]:
            raise ValueError(
                "denominator must have shape [batch, gaussian], got "
                f"{tuple(denominator.shape)}"
            )

        use_cuda_timing = pixel_error.is_cuda
        if use_cuda_timing:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
        else:
            start_time = perf_counter()
        numerator_chunks = []
        per_view_chunks = []
        chunk_times: list[float] = []
        chunk_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for chunk in pixel_error.split(chunk_size, dim=2):
            if use_cuda_timing:
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
            else:
                chunk_start = perf_counter()
            probe_result = self._run_gaussian_probe(
                gaussians,
                extrinsics,
                intrinsics,
                near,
                far,
                image_shape,
                pixel_cue=chunk,
                include_contribution=False,
            )
            numerator_chunks.append(probe_result["cue_sum"])
            per_view_chunks.append(probe_result["per_view_cue_sum"])
            if use_cuda_timing:
                event_end.record()
                chunk_events.append((event_start, event_end))
            else:
                chunk_times.append(perf_counter() - chunk_start)

        if use_cuda_timing:
            total_end.record()
            total_end.synchronize()
            elapsed_seconds = total_start.elapsed_time(total_end) / 1000.0
            chunk_times = [
                start.elapsed_time(end) / 1000.0 for start, end in chunk_events
            ]
        else:
            elapsed_seconds = perf_counter() - start_time

        numerator = torch.cat(numerator_chunks, dim=-1)
        per_view_numerator = torch.cat(per_view_chunks, dim=-1)
        if numerator.shape[-1] != channels:
            raise RuntimeError(
                f"Lifted {numerator.shape[-1]} channels from a {channels}-channel cue"
            )
        return {
            "cue_sum": numerator,
            "per_view_cue_sum": per_view_numerator,
            "cue": self._normalize_cue(numerator, denominator.float(), eps),
            "elapsed_seconds": numerator.new_tensor(elapsed_seconds),
            "mean_chunk_seconds": numerator.new_tensor(
                sum(chunk_times) / max(len(chunk_times), 1)
            ),
            "num_chunks": numerator.new_tensor(len(numerator_chunks), dtype=torch.int64),
        }

    def compute_gaussian_error_cues(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            feature_error: Tensor,
            rgb_error: Tensor,
            chunk_size: int = 32,
            max_probe_channels: int = 34,
            eps: float = 1e-8,
    ) -> dict[str, Tensor]:
        """Lift full feature and RGB errors with one shared denominator."""
        if max_probe_channels < 4 or max_probe_channels > 34:
            raise ValueError(
                "Combined RGB/contribution lifting requires max_probe_channels "
                f"in [4, 34], got {max_probe_channels}"
            )
        if rgb_error.shape[2] != 3:
            raise ValueError(f"rgb_error must have 3 channels, got {rgb_error.shape[2]}")
        rgb_result = self.compute_gaussian_contribution(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            pixel_cue=rgb_error,
            eps=eps,
        )
        feature_result = self.lift_pixel_error_in_chunks(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            feature_error,
            denominator=rgb_result["total"],
            chunk_size=chunk_size,
            max_probe_channels=max_probe_channels,
            eps=eps,
        )
        return {
            "feature_cue": feature_result["cue"],
            "feature_cue_sum": feature_result["cue_sum"],
            "per_view_feature_cue_sum": feature_result["per_view_cue_sum"],
            "rgb_cue": rgb_result["cue"],
            "rgb_cue_sum": rgb_result["cue_sum"],
            "per_view_rgb_cue_sum": rgb_result["per_view_cue_sum"],
            "per_view": rgb_result["per_view"],
            "total": rgb_result["total"],
            "radii": rgb_result["radii"],
            "rendered_alpha": rgb_result["rendered_alpha"],
            "feature_lifting_seconds": feature_result["elapsed_seconds"],
            "feature_chunk_seconds": feature_result["mean_chunk_seconds"],
            "feature_probe_count": feature_result["num_chunks"],
        }




