from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...dataset.shims.normalize_shim import normalize_image
from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam
from ..super_resolution import FrozenSwinIRUpsampler
from ..types import Gaussians
from .common.gaussians import build_covariance, quaternion_to_matrix


_PTV3_PARENT = Path("/space0/mengxl")
if str(_PTV3_PARENT) not in sys.path:
    sys.path.insert(0, str(_PTV3_PARENT))

from PointTransformerV3.model import Block, Point, PointSequential  # noqa: E402


@dataclass
class SR3RCfg:
    enabled: bool = False
    upscale: int = 4
    swinir_weight_path: str = (
        "/space0/mengxl/NAS3R-master/pretrained_weights/"
        "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    )
    encoder_dim: int = 1024
    decoder_dim: int = 768
    token_heads: int = 8
    token_dropout: float = 0.0
    shuffle_beta: float = 0.5
    shuffle_axis_divisor: float = 4.0
    shuffle_lateral_divisor: float = 1.9
    shuffle_opacity_threshold: float = 0.5
    apply_shuffle_opacity_threshold: bool = True
    ptv3_channels: int = 768
    ptv3_depth: int = 4
    ptv3_heads: int = 8
    ptv3_patch_size: int = 1024
    ptv3_mlp_ratio: float = 4.0
    ptv3_enable_flash: bool = True
    ptv3_grid_depth: int = 15
    debug_interval: int = 500


@dataclass
class GaussianShuffleResult:
    gaussians: Gaussians
    source_view: torch.Tensor
    parent_index: torch.Tensor
    split_mask: torch.Tensor


def detach_gaussians(gaussians: Gaussians) -> Gaussians:
    return Gaussians(
        means=gaussians.means.detach(),
        covariances=gaussians.covariances.detach(),
        rotations=gaussians.rotations.detach(),
        scales=gaussians.scales.detach(),
        harmonics=gaussians.harmonics.detach(),
        opacities=gaussians.opacities.detach(),
    )


def compose_gaussian_residuals(
    dense: Gaussians,
    delta_means: torch.Tensor,
    delta_opacities: torch.Tensor,
    delta_rotations: torch.Tensor,
    delta_scales: torch.Tensor,
    delta_harmonics: torch.Tensor,
    eps: float = 1e-6,
) -> Gaussians:
    """Compose residuals in the native parameter domain of each attribute.

    NAS3R exposes activated opacity and positive linear scale, whereas
    AnchorSplat updates opacity logits and log-scales. Convert the detached
    template back to those unconstrained domains before residual addition, then
    map it to the representation expected by the NAS3R rasterizer.
    """
    means = dense.means + delta_means
    opacity_logits = torch.logit(dense.opacities.clamp(eps, 1 - eps))
    opacities = torch.sigmoid(opacity_logits + delta_opacities)
    rotation_candidate = dense.rotations + delta_rotations
    rotation_norm = rotation_candidate.norm(dim=-1, keepdim=True)
    normalized_rotations = rotation_candidate / rotation_norm.clamp_min(eps)
    identity_rotation = torch.zeros_like(rotation_candidate)
    identity_rotation[..., 3] = 1
    rotations = torch.where(
        rotation_norm > eps,
        normalized_rotations,
        identity_rotation,
    )
    log_scales = dense.scales.clamp_min(eps).log()
    scales = torch.exp(log_scales + delta_scales)
    harmonics = dense.harmonics + delta_harmonics
    return Gaussians(
        means=means,
        covariances=build_covariance(scales, rotations),
        rotations=rotations,
        scales=scales,
        harmonics=harmonics,
        opacities=opacities,
    )


class GaussianShuffleSplit(nn.Module):
    """The six-child Gaussian Shuffle Split used by SR3R/S2Gaussian.

    A fixed six children per parent keeps the batched representation dense. When
    opacity thresholding is enabled, low-opacity parents are repeated without a
    geometric split; high-opacity parents receive the paper's principal-axis split.
    """

    def __init__(self, cfg: SR3RCfg) -> None:
        super().__init__()
        self.cfg = cfg
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("directions", directions, persistent=False)

        scale_factors = torch.full(
            (6, 3),
            1.0 / cfg.shuffle_lateral_divisor,
            dtype=torch.float32,
        )
        scale_factors[0:2, 0] = 1.0 / cfg.shuffle_axis_divisor
        scale_factors[2:4, 1] = 1.0 / cfg.shuffle_axis_divisor
        scale_factors[4:6, 2] = 1.0 / cfg.shuffle_axis_divisor
        self.register_buffer("scale_factors", scale_factors, persistent=False)

    def forward(self, gaussians: Gaussians, num_views: int) -> GaussianShuffleResult:
        gaussians = detach_gaussians(gaussians)
        b, num_parents, _ = gaussians.means.shape
        if num_parents % num_views != 0:
            raise ValueError(
                f"Expected {num_parents} Gaussians to divide into {num_views} source views."
            )

        rotations = quaternion_to_matrix(gaussians.rotations)
        local_offsets = (
            gaussians.scales[:, :, None]
            * self.directions[None, None].to(gaussians.scales)
            * self.cfg.shuffle_beta
        )
        world_offsets = torch.einsum("bnij,bnkj->bnki", rotations, local_offsets)

        if self.cfg.apply_shuffle_opacity_threshold:
            split_mask = gaussians.opacities > self.cfg.shuffle_opacity_threshold
        else:
            split_mask = torch.ones_like(gaussians.opacities, dtype=torch.bool)

        child_means = gaussians.means[:, :, None] + (
            world_offsets * split_mask[:, :, None, None]
        )
        split_scales = (
            gaussians.scales[:, :, None]
            * self.scale_factors[None, None].to(gaussians.scales)
        )
        child_scales = torch.where(
            split_mask[:, :, None, None],
            split_scales,
            gaussians.scales[:, :, None].expand(-1, -1, 6, -1),
        )
        child_rotations = gaussians.rotations[:, :, None].expand(-1, -1, 6, -1)
        child_harmonics = gaussians.harmonics[:, :, None].expand(-1, -1, 6, -1, -1)
        child_opacities = gaussians.opacities[:, :, None].expand(-1, -1, 6)

        child_means = rearrange(child_means, "b n k xyz -> b (n k) xyz")
        child_scales = rearrange(child_scales, "b n k xyz -> b (n k) xyz")
        child_rotations = rearrange(child_rotations, "b n k q -> b (n k) q")
        child_harmonics = rearrange(
            child_harmonics,
            "b n k rgb sh -> b (n k) rgb sh",
        )
        child_opacities = rearrange(child_opacities, "b n k -> b (n k)")

        parents_per_view = num_parents // num_views
        source_view = torch.arange(
            num_views,
            device=gaussians.means.device,
            dtype=torch.long,
        ).repeat_interleave(parents_per_view * 6)
        source_view = source_view[None].expand(b, -1)
        parent_index = torch.arange(
            num_parents,
            device=gaussians.means.device,
            dtype=torch.long,
        ).repeat_interleave(6)
        parent_index = parent_index[None].expand(b, -1)

        dense = Gaussians(
            means=child_means,
            covariances=build_covariance(child_scales, child_rotations),
            rotations=child_rotations,
            scales=child_scales,
            harmonics=child_harmonics,
            opacities=child_opacities,
        )
        return GaussianShuffleResult(
            gaussians=dense,
            source_view=source_view,
            parent_index=parent_index,
            split_mask=split_mask,
        )


class BidirectionalTokenFusion(nn.Module):
    """Equation (4) in SR3R, applied before any multi-view decoding."""

    def __init__(self, channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.hr_norm = nn.LayerNorm(channels)
        self.lr_norm = nn.LayerNorm(channels)
        self.hr_queries_lr = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.lr_queries_hr = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fuse = nn.Linear(2 * channels, channels)
        self.output_norm = nn.LayerNorm(channels)

    @staticmethod
    def align_lr_tokens(
        lr_tokens: torch.Tensor,
        lr_grid: tuple[int, int],
        hr_grid: tuple[int, int],
    ) -> torch.Tensor:
        if lr_grid == hr_grid:
            return lr_tokens
        b, v, n, c = lr_tokens.shape
        if n != lr_grid[0] * lr_grid[1]:
            raise ValueError(
                f"LR token count {n} does not match token grid {lr_grid}."
            )
        lr_map = rearrange(
            lr_tokens,
            "b v (h w) c -> (b v) c h w",
            h=lr_grid[0],
            w=lr_grid[1],
        )
        lr_map = F.interpolate(
            lr_map,
            size=hr_grid,
            mode="bilinear",
            align_corners=False,
        )
        return rearrange(
            lr_map,
            "(b v) c h w -> b v (h w) c",
            b=b,
            v=v,
        )

    def forward(
        self,
        lr_tokens: torch.Tensor,
        hr_tokens: torch.Tensor,
        lr_grid: tuple[int, int],
        hr_grid: tuple[int, int],
    ) -> torch.Tensor:
        lr_tokens = self.align_lr_tokens(lr_tokens, lr_grid, hr_grid)
        if lr_tokens.shape != hr_tokens.shape:
            raise ValueError(
                "Aligned LR and HR tokens must share shape, got "
                f"{tuple(lr_tokens.shape)} and {tuple(hr_tokens.shape)}."
            )
        b, v, n, c = hr_tokens.shape
        hr = rearrange(self.hr_norm(hr_tokens), "b v n c -> (b v) n c")
        lr = rearrange(self.lr_norm(lr_tokens), "b v n c -> (b v) n c")
        hr_from_lr = self.hr_queries_lr(hr, lr, lr, need_weights=False)[0]
        lr_from_hr = self.lr_queries_hr(lr, hr, hr, need_weights=False)[0]
        fused = self.output_norm(self.fuse(torch.cat([hr_from_lr, lr_from_hr], dim=-1)))
        return rearrange(fused, "(b v) n c -> b v n c", b=b, v=v, n=n, c=c)


class GaussianOffsetPTV3(nn.Module):
    """Projects dense Gaussians into t_de and predicts unconstrained residuals."""

    def __init__(self, cfg: SR3RCfg, sh_degree: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.sh_dim = (sh_degree + 1) ** 2
        channels = cfg.ptv3_channels

        self.image_projection = nn.Sequential(
            nn.LayerNorm(cfg.decoder_dim),
            nn.Linear(cfg.decoder_dim, channels),
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(12, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.input_norm = nn.LayerNorm(channels)

        order = ("z", "z-trans", "hilbert", "hilbert-trans")
        self.order = order
        self.blocks = PointSequential()
        for index in range(cfg.ptv3_depth):
            self.blocks.add(
                Block(
                    channels=channels,
                    num_heads=cfg.ptv3_heads,
                    patch_size=cfg.ptv3_patch_size,
                    mlp_ratio=cfg.ptv3_mlp_ratio,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    pre_norm=True,
                    order_index=index % len(order),
                    cpe_indice_key="sr3r_ptv3",
                    enable_rpe=False,
                    enable_flash=cfg.ptv3_enable_flash,
                    upcast_attention=not cfg.ptv3_enable_flash,
                    upcast_softmax=not cfg.ptv3_enable_flash,
                ),
                name=f"block{index}",
            )

        output_dim = 3 + 1 + 4 + 3 + 3 * self.sh_dim
        self.pre_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, output_dim),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _sample_source_features(
        means: torch.Tensor,
        source_view: torch.Tensor,
        feature_maps: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_points, _ = means.shape
        _, num_views, channels, _, _ = feature_maps.shape
        sampled = means.new_zeros((b, num_points, channels), dtype=feature_maps.dtype)
        sampled_intrinsics = means.new_zeros((b, num_points, 9))

        for view_index in range(num_views):
            indices = torch.nonzero(
                source_view[0] == view_index,
                as_tuple=False,
            ).squeeze(-1)
            if indices.numel() == 0:
                continue
            if not torch.equal(
                source_view[:, indices],
                torch.full_like(source_view[:, indices], view_index),
            ):
                raise ValueError("source_view layout must be identical across the batch.")

            view_means = means[:, indices]
            camera_points = transform_world2cam(
                homogenize_points(view_means),
                extrinsics[:, view_index, None],
            )[..., :3]
            projected = project_camera_space(
                camera_points,
                intrinsics[:, view_index, None],
            )
            valid = (
                (camera_points[..., 2] > 1e-6)
                & (projected[..., 0] >= 0)
                & (projected[..., 0] <= 1)
                & (projected[..., 1] >= 0)
                & (projected[..., 1] <= 1)
            )
            grid = projected.mul(2).sub(1).unsqueeze(2)
            view_features = F.grid_sample(
                feature_maps[:, view_index],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            view_features = rearrange(view_features, "b c n 1 -> b n c")
            sampled[:, indices] = view_features * valid[..., None].to(view_features.dtype)
            sampled_intrinsics[:, indices] = rearrange(
                intrinsics[:, view_index],
                "b i j -> b 1 (i j)",
            ).expand(-1, indices.numel(), -1)
        return sampled, sampled_intrinsics

    def _grid_coordinates(self, means: torch.Tensor) -> torch.Tensor:
        coords = means.detach()
        coord_min = coords.amin(dim=1, keepdim=True)
        extent = (coords.amax(dim=1, keepdim=True) - coord_min).amax(
            dim=-1,
            keepdim=True,
        )
        normalized = (coords - coord_min) / extent.clamp_min(1e-6)
        grid_max = (1 << self.cfg.ptv3_grid_depth) - 1
        return (normalized * grid_max).floor().clamp(0, grid_max).int()

    @staticmethod
    def _normalize_means(means: torch.Tensor) -> torch.Tensor:
        center = means.mean(dim=1, keepdim=True)
        radius = (means - center).norm(dim=-1).amax(dim=1, keepdim=True)
        return (means - center) / radius[..., None].clamp_min(1e-6)

    def forward(
        self,
        template: GaussianShuffleResult,
        decoded_feature_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor | Gaussians]:
        dense = template.gaussians
        b, num_points, _ = dense.means.shape
        queried, point_intrinsics = self._sample_source_features(
            dense.means,
            template.source_view,
            decoded_feature_map,
            extrinsics,
            intrinsics,
        )
        position_input = torch.cat(
            [self._normalize_means(dense.means), point_intrinsics],
            dim=-1,
        )
        features = self.input_norm(
            self.image_projection(queried) + self.position_embedding(position_input)
        )

        point = Point(
            {
                "coord": rearrange(dense.means, "b n xyz -> (b n) xyz"),
                "grid_coord": rearrange(
                    self._grid_coordinates(dense.means),
                    "b n xyz -> (b n) xyz",
                ),
                "feat": rearrange(features, "b n c -> (b n) c"),
                "offset": torch.arange(
                    1,
                    b + 1,
                    device=dense.means.device,
                    dtype=torch.long,
                )
                * num_points,
            }
        )
        point.serialization(order=self.order, shuffle_orders=True)
        point.sparsify()
        point = self.blocks(point)
        features = rearrange(point.feat, "(b n) c -> b n c", b=b, n=num_points)

        delta = self.delta_head(self.pre_head(features))
        delta_means, delta_opacities, delta_rotations, delta_scales, delta_harmonics = (
            delta.split((3, 1, 4, 3, 3 * self.sh_dim), dim=-1)
        )
        delta_harmonics = rearrange(
            delta_harmonics,
            "b n (rgb sh) -> b n rgb sh",
            rgb=3,
            sh=self.sh_dim,
        )

        # Means and SH use direct residuals. Following AnchorSplat's native
        # parameterization, opacity and scale are updated in logit/log space,
        # and the additive quaternion residual is normalized afterwards.
        refined = compose_gaussian_residuals(
            dense,
            delta_means,
            delta_opacities.squeeze(-1),
            delta_rotations,
            delta_scales,
            delta_harmonics,
        )
        return {
            "gaussians": refined,
            "delta_means": delta_means,
            "delta_opacities": delta_opacities.squeeze(-1),
            "delta_rotations": delta_rotations,
            "delta_scales": delta_scales,
            "delta_harmonics": delta_harmonics,
        }


class SR3RMapping(nn.Module):
    def __init__(self, cfg: SR3RCfg, sh_degree: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.upsampler = FrozenSwinIRUpsampler(
            cfg.swinir_weight_path,
            upscale=cfg.upscale,
            img_size=64,
            window_size=8,
        )
        self.token_fusion = BidirectionalTokenFusion(
            cfg.encoder_dim,
            cfg.token_heads,
            cfg.token_dropout,
        )
        self.shuffle = GaussianShuffleSplit(cfg)
        self.offset_refiner = GaussianOffsetPTV3(cfg, sh_degree)

    def forward(
        self,
        backbone: nn.Module,
        base_gaussians: Gaussians,
        lr_encoder_tokens: torch.Tensor,
        lr_image_shape: tuple[int, int],
        context_image_lr: torch.Tensor,
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
    ) -> dict[str, object]:
        sr_images = self.upsampler(context_image_lr)
        hr_context = {
            "image": normalize_image(sr_images),
            "intrinsics": context_intrinsics,
        }
        with torch.no_grad():
            hr_encoded = backbone.encode_views(hr_context)
        hr_tokens = hr_encoded["image_tokens"].detach()

        patch_size = int(backbone.patch_size)
        lr_grid = (
            lr_image_shape[0] // patch_size,
            lr_image_shape[1] // patch_size,
        )
        hr_grid = (
            sr_images.shape[-2] // patch_size,
            sr_images.shape[-1] // patch_size,
        )
        fused_tokens = self.token_fusion(
            lr_encoder_tokens.detach(),
            hr_tokens,
            lr_grid,
            hr_grid,
        )

        # The original frozen NAS3R decoder now performs the first multi-view
        # interaction. Gradients flow through its operations into fused_tokens, but
        # not into the frozen decoder parameters.
        decoded = backbone.decode_views(
            hr_encoded,
            feature_override=fused_tokens,
            target_num_views=0,
        )
        decoded_tokens = decoded["dec_feat"][-1]
        decoded_feature_map = rearrange(
            decoded_tokens,
            "b v (h w) c -> b v c h w",
            h=hr_grid[0],
            w=hr_grid[1],
        )

        dense_template = self.shuffle(
            detach_gaussians(base_gaussians),
            num_views=context_image_lr.shape[1],
        )
        refined = self.offset_refiner(
            dense_template,
            decoded_feature_map,
            context_extrinsics,
            context_intrinsics,
        )
        return {
            **refined,
            "base_gaussians": detach_gaussians(base_gaussians),
            "dense_gaussians": dense_template.gaussians,
            "shuffle_split_mask": dense_template.split_mask,
            "sr_images": sr_images.detach(),
            "lr_encoder_tokens": lr_encoder_tokens.detach(),
            "hr_encoder_tokens": hr_tokens,
            "fused_encoder_tokens": fused_tokens,
            "decoded_tokens": decoded_tokens,
        }
