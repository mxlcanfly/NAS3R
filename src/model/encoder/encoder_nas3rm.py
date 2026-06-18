from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import math

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .common.gaussians import build_covariance
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector, \
    unproject_depth_map_to_point_map_batch
from .heads.pose_head import PoseHeadCfg
from .multi_view_consistency import FeatureMultiViewConsistencyEstimator
from .point_offset_decoder import PointOffsetDecoder
from .resunet_fusion import HiSplatResUnetTokenFusion
from ..super_resolution import FrozenSwinIRUpsampler
from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNAS3RMCfg:
    name: Literal["nas3r-m"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pose_head: PoseHeadCfg

    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True
    pose_head_type: str = 'mlp'
    estimating_focal: bool = False
    estimating_pose: bool = True

    depth_activation: str = 'sigmoid'

    equal_fxfy: bool = True
    equal_view_intrinsics: bool = True
    use_swinir: bool = False
    swinir_weights: str = ""
    swinir_upscale: int = 4
    swinir_input_size: int = 64
    use_resunet_token_fusion: bool = False
    unimatch_weights_path: str = ""
    use_image_only_feature_consistency: bool = False
    image_only_consistency_patch_size: int = 1
    image_only_consistency_depth_relative_tolerance: float = 0.1
    use_anchor_multiview_feature_aggregation: bool = False
    anchor_feature_occlusion_tau: float = 10.0
    anchor_feature_similarity_beta: float = 1.0
    anchor_feature_eps: float = 1e-6
    anchor_feature_patch_size: int = 4
    use_point_offset_decoder: bool = False
    point_offset_hidden_dim: int = 128
    point_offset_depth: int = 2
    point_offset_num_heads: int = 8
    point_offset_patch_size: int = 48
    point_offset_k: int = 8
    point_offset_grid_size: float = 0.02
    point_offset_scale: float = 0.1
    use_child_gaussian_residual: bool = False
    child_feature_patch_size: int = 2
    child_gaussian_hidden_dim: int = 256
    child_gaussian_fourier_frequencies: int = 6
    child_gaussian_enable_absolute_pe: bool = True
    child_scale_divisor: float = 2.0


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNAS3RM(Encoder[EncoderNAS3RMCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNAS3RMCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.swinir = None
        if cfg.use_swinir:
            if not cfg.swinir_weights:
                raise ValueError("swinir_weights must be set when use_swinir=True")
            self.swinir = FrozenSwinIRUpsampler(
                weight_path=cfg.swinir_weights,
                upscale=cfg.swinir_upscale,
                img_size=cfg.swinir_input_size,
            )
        self.resunet_token_fusion = None
        if cfg.use_resunet_token_fusion:
            self.resunet_token_fusion = HiSplatResUnetTokenFusion(
                token_dim=self.backbone.dec_embed_dim,
            )
            if cfg.unimatch_weights_path:
                self.resunet_token_fusion.load_unimatch_encoder(cfg.unimatch_weights_path)
        self.image_only_consistency = None
        if cfg.use_image_only_feature_consistency:
            self.image_only_consistency = FeatureMultiViewConsistencyEstimator(
                patch_size=cfg.image_only_consistency_patch_size,
                depth_relative_tolerance=cfg.image_only_consistency_depth_relative_tolerance,
            )
        self.anchor_feature_dim = cfg.anchor_feature_patch_size ** 2 * (32 + 3)
        self.point_offset_decoder = None
        if cfg.use_point_offset_decoder:
            self.point_offset_decoder = PointOffsetDecoder(
                in_channels=self.anchor_feature_dim,
                hidden_channels=cfg.point_offset_hidden_dim,
                depth=cfg.point_offset_depth,
                num_heads=cfg.point_offset_num_heads,
                patch_size=cfg.point_offset_patch_size,
                k_offsets=cfg.point_offset_k,
                grid_size=cfg.point_offset_grid_size,
                offset_scale=cfg.point_offset_scale,
            )
        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.child_gaussian_head = None
        if cfg.use_child_gaussian_residual:
            child_obs_dim = cfg.child_feature_patch_size ** 2 * (32 + 3)
            offset_pe_dim = 3 * 2 * cfg.child_gaussian_fourier_frequencies
            child_gaussian_param_dim = 3 + 4 + 1 + 3 * (self.gaussian_adapter.d_sh)
            child_input_dim = (
                child_obs_dim
                + cfg.point_offset_hidden_dim
                + offset_pe_dim
                + child_gaussian_param_dim
            )
            child_output_dim = child_gaussian_param_dim
            self.child_gaussian_head = nn.Sequential(
                nn.Linear(child_input_dim, cfg.child_gaussian_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.child_gaussian_hidden_dim, cfg.child_gaussian_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.child_gaussian_hidden_dim, cfg.child_gaussian_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.child_gaussian_hidden_dim, child_output_dim),
            )
            nn.init.zeros_(self.child_gaussian_head[-1].weight)
            nn.init.zeros_(self.child_gaussian_head[-1].bias)
            self.register_buffer(
                "child_offset_frequencies",
                2.0 ** torch.arange(cfg.child_gaussian_fourier_frequencies),
                persistent=False,
            )

        self.patch_size = self.backbone.patch_embed.patch_size[0]

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # base (1 for opacity)

        self.gs_params_head_type = cfg.gs_params_head_type

        if self.cfg.depth_activation == 'exp':
            self.set_depth_head(output_mode='depth', head_type='dpt', landscape_only=True,
                                depth_mode=('exp', -inf, inf), conf_mode=None, )
        elif self.cfg.depth_activation == 'sigmoid':
            self.set_depth_head(output_mode='depth', head_type='dpt', landscape_only=True,
                                depth_mode=('range', 1, 100.), conf_mode=None, )
        else:
            raise NotImplementedError

        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        if self.cfg.estimating_pose:
            self.set_pose_head(cfg, cfg.pose_head_type)

    def set_depth_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_depth_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_depth_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.depth_head1 = transpose_to_landscape(self.downstream_depth_head1, activate=landscape_only)
        self.depth_head2 = transpose_to_landscape(self.downstream_depth_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)

        elif 'dpt' in head_type:
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                    out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False,
                                                     out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def set_pose_head(self, cfg, head_type='mlp'):
        self.pose_head = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)
        self.pose_head2 = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)

    def map_pdf_to_opacity(
            self,
            pdf: Float[Tensor, " *batch"],
            global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2 ** x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_depth_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'depth_head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def _super_resolve(self, images: Tensor) -> Tensor:
        if self.swinir is None:
            raise RuntimeError("_super_resolve called while use_swinir=False")
        expected_size = (self.cfg.swinir_input_size, self.cfg.swinir_input_size)
        if images.shape[-2:] != expected_size:
            raise ValueError(
                f"SwinIR expects {expected_size} inputs, got {tuple(images.shape[-2:])}"
            )
        return self.swinir(images)

    @staticmethod
    def _sample_projected_map(
            feature_map: Tensor,
            projected_xy: Tensor,
            patch_size: int,
    ) -> Tensor:
        b, num_views = feature_map.shape[:2]
        _, _, channels, height, width = feature_map.shape
        radius = (patch_size - 1) / 2
        offsets = torch.arange(
            patch_size,
            device=feature_map.device,
            dtype=feature_map.dtype,
        ) - radius
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        pixel_scale = torch.tensor(
            (max(width - 1, 1), max(height - 1, 1)),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        patch_offsets = torch.stack((xx, yy), dim=-1).reshape(-1, 2) / pixel_scale
        patch_xy = projected_xy[:, :, :, None] + patch_offsets
        sampled = F.grid_sample(
            rearrange(feature_map, "b v c h w -> (b v) c h w"),
            rearrange(
                patch_xy * 2 - 1,
                "b n v p xy -> (b v) n p xy",
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return rearrange(
            sampled,
            "(b v) c n p -> b n v (p c)",
            b=b,
            v=num_views,
            c=channels,
        )

    def _compute_multiview_weights(
            self,
            consistency,
            source_view_indices: Tensor,
            num_views: int,
    ) -> dict[str, Tensor]:
        source_view_mask = F.one_hot(
            source_view_indices.long(),
            num_classes=num_views,
        ).bool()
        projection_valid = consistency.view_projection_valid
        valid_mask = projection_valid | source_view_mask

        w_sim = consistency.per_view_consistency_weight.clamp(0, 1)
        w_sim = torch.where(source_view_mask, torch.ones_like(w_sim), w_sim)
        w_occ = torch.sigmoid(
            -self.cfg.anchor_feature_occlusion_tau * consistency.occlusion_delta
        )
        w_occ = torch.where(
            consistency.occlusion_valid_mask,
            w_occ,
            torch.zeros_like(w_occ),
        )
        w_occ = torch.where(source_view_mask, torch.ones_like(w_occ), w_occ)

        log_weight = (
            torch.log(w_occ.clamp_min(self.cfg.anchor_feature_eps))
            + self.cfg.anchor_feature_similarity_beta
            * torch.log(w_sim.clamp_min(self.cfg.anchor_feature_eps))
        )
        log_weight = log_weight.masked_fill(~valid_mask, -torch.finfo(log_weight.dtype).max)
        weights = torch.softmax(log_weight, dim=-1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.cfg.anchor_feature_eps)

        return {
            "weights": weights,
            "valid_mask": valid_mask,
            "source_view_mask": source_view_mask,
            "w_occ": w_occ,
            "w_sim": w_sim,
        }

    def _aggregate_anchor_multiview_features(
            self,
            anchors: Tensor,
            feature_map: Tensor,
            sr_images: Tensor,
            consistency,
            extrinsics: Tensor,
            intrinsics: Tensor,
            source_view_indices: Tensor,
    ) -> dict[str, Tensor]:
        b, num_views = extrinsics.shape[:2]
        num_anchors = anchors.shape[1]
        anchors_per_view = repeat(
            anchors,
            "b n xyz -> b v n xyz",
            v=num_views,
        )
        camera_points = transform_world2cam(
            homogenize_points(anchors_per_view),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = camera_points[..., -1]
        projected_xy = project_camera_space(
            camera_points,
            intrinsics[:, :, None],
        )
        projected_xy = rearrange(projected_xy, "b v n xy -> b n v xy")
        camera_depth = rearrange(camera_depth, "b v n -> b n v")

        sampled_features = self._sample_projected_map(
            feature_map,
            projected_xy,
            self.cfg.anchor_feature_patch_size,
        )
        sampled_rgb = self._sample_projected_map(
            sr_images,
            projected_xy,
            self.cfg.anchor_feature_patch_size,
        )

        weight_dict = self._compute_multiview_weights(
            consistency,
            source_view_indices,
            num_views,
        )
        weights = weight_dict["weights"]

        aggregated_features = (sampled_features * weights[..., None]).sum(dim=2)
        aggregated_rgb = (sampled_rgb * weights[..., None]).sum(dim=2)

        if source_view_indices.shape != (b, num_anchors):
            raise ValueError(
                f"Expected source_view_indices shape {(b, num_anchors)}, got "
                f"{tuple(source_view_indices.shape)}."
            )

        return {
            "feature_3d": torch.cat([aggregated_features, aggregated_rgb], dim=-1),
            "weights": weights,
            "valid_mask": weight_dict["valid_mask"],
            "w_occ": weight_dict["w_occ"],
            "w_sim": weight_dict["w_sim"],
        }

    def _fourier_encode_positions(self, positions: Tensor) -> Tensor:
        if self.cfg.child_gaussian_fourier_frequencies <= 0:
            return positions.new_zeros((*positions.shape[:-1], 0))
        frequencies = self.child_offset_frequencies.to(
            device=positions.device,
            dtype=positions.dtype,
        )
        encoded = positions[..., None, :] * frequencies.view(1, 1, 1, -1, 1)
        encoded = torch.cat([encoded.sin(), encoded.cos()], dim=-2)
        return rearrange(encoded, "b n k two_f xyz -> b n k (two_f xyz)")

    def _aggregate_child_multiview_features(
            self,
            child_centers: Tensor,
            feature_map: Tensor,
            sr_images: Tensor,
            parent_weights: Tensor,
            extrinsics: Tensor,
            intrinsics: Tensor,
    ) -> dict[str, Tensor]:
        b, num_anchors, k_offsets, _ = child_centers.shape
        num_views = extrinsics.shape[1]
        child_centers_flat = rearrange(child_centers, "b n k xyz -> b (n k) xyz")
        child_centers_per_view = repeat(
            child_centers_flat,
            "b nk xyz -> b v nk xyz",
            v=num_views,
        )
        camera_points = transform_world2cam(
            homogenize_points(child_centers_per_view),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = camera_points[..., -1]
        projected_xy = project_camera_space(
            camera_points,
            intrinsics[:, :, None],
        )
        projected_xy = rearrange(projected_xy, "b v nk xy -> b nk v xy")
        camera_depth = rearrange(camera_depth, "b v nk -> b nk v")

        sampled_features = self._sample_projected_map(
            feature_map,
            projected_xy,
            self.cfg.child_feature_patch_size,
        )
        sampled_rgb = self._sample_projected_map(
            sr_images,
            projected_xy,
            self.cfg.child_feature_patch_size,
        )

        projection_valid = (
            (camera_depth > 0)
            & projected_xy.isfinite().all(dim=-1)
            & (projected_xy >= 0).all(dim=-1)
            & (projected_xy <= 1).all(dim=-1)
        )
        child_weights = repeat(
            parent_weights,
            "b n v -> b n k v",
            k=k_offsets,
        )
        child_weights = rearrange(child_weights, "b n k v -> b (n k) v")
        child_weights = torch.where(
            projection_valid,
            child_weights,
            torch.zeros_like(child_weights),
        )
        child_weights = child_weights / child_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.cfg.anchor_feature_eps)

        aggregated_features = (sampled_features * child_weights[..., None]).sum(dim=2)
        aggregated_rgb = (sampled_rgb * child_weights[..., None]).sum(dim=2)
        feature_3d = torch.cat([aggregated_features, aggregated_rgb], dim=-1)

        return {
            "feature_3d": rearrange(feature_3d, "b (n k) c -> b n k c", n=num_anchors, k=k_offsets),
            "weights": rearrange(child_weights, "b (n k) v -> b n k v", n=num_anchors, k=k_offsets),
            "valid_mask": rearrange(projection_valid, "b (n k) v -> b n k v", n=num_anchors, k=k_offsets),
        }

    def _decode_child_gaussian_residuals(
            self,
            parent_gaussians,
            point_offset_densification: dict[str, Tensor],
            anchor_multiview_features: dict[str, Tensor],
            feature_map: Tensor,
            sr_images: Tensor,
            extrinsics: Tensor,
            intrinsics: Tensor,
    ) -> dict[str, Tensor | Gaussians]:
        if self.child_gaussian_head is None:
            raise RuntimeError("_decode_child_gaussian_residuals called while disabled")

        child_centers = point_offset_densification["child_centers"]
        offsets = point_offset_densification["offsets"]
        b, num_anchors, k_offsets, _ = child_centers.shape

        child_multiview_features = self._aggregate_child_multiview_features(
            child_centers,
            feature_map,
            sr_images,
            anchor_multiview_features["weights"],
            extrinsics,
            intrinsics,
        )
        parent_point_features = repeat(
            point_offset_densification["parent_features"],
            "b n c -> b n k c",
            k=k_offsets,
        )
        parent_scales = rearrange(
            parent_gaussians.scales,
            "b v r srf spp xyz -> b (v r srf spp) xyz",
        ).detach()
        parent_rotations = rearrange(
            parent_gaussians.rotations,
            "b v r srf spp xyzw -> b (v r srf spp) xyzw",
        ).detach()
        parent_harmonics = rearrange(
            parent_gaussians.harmonics,
            "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
        ).detach()
        parent_opacities_raw = torch.logit(
            rearrange(
                parent_gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ).detach().clamp(1e-6, 1 - 1e-6)
            / k_offsets,
            eps=1e-6,
        )
        base_scales = repeat(
            parent_scales / self.cfg.child_scale_divisor,
            "b n xyz -> b n k xyz",
            k=k_offsets,
        )
        base_rotations = repeat(
            parent_rotations,
            "b n xyzw -> b n k xyzw",
            k=k_offsets,
        )
        base_harmonics = repeat(
            parent_harmonics,
            "b n c d_sh -> b n k c d_sh",
            k=k_offsets,
        )
        base_opacities_raw = repeat(
            parent_opacities_raw,
            "b n -> b n k",
            k=k_offsets,
        )[..., None]
        base_gaussian_features = torch.cat(
            [
                base_scales,
                base_rotations,
                base_opacities_raw,
                rearrange(base_harmonics, "b n k c d_sh -> b n k (c d_sh)"),
            ],
            dim=-1,
        )
        positions_for_encoding = child_centers if self.cfg.child_gaussian_enable_absolute_pe else offsets
        position_encoding = self._fourier_encode_positions(positions_for_encoding)
        child_input = torch.cat(
            [
                child_multiview_features["feature_3d"],
                parent_point_features,
                position_encoding,
                base_gaussian_features,
            ],
            dim=-1,
        )
        child_input = rearrange(child_input, "b n k c -> b (n k) c")
        delta_gaussians = self.child_gaussian_head(child_input)

        sh_dim = 3 * self.gaussian_adapter.d_sh
        delta_scales, delta_rotations, delta_opacities, delta_shs = (
            delta_gaussians.split((3, 4, 1, sh_dim), dim=-1)
        )

        child_means = rearrange(child_centers, "b n k xyz -> b (n k) xyz")
        child_scales = (
            rearrange(base_scales, "b n k xyz -> b (n k) xyz") + delta_scales
        ).clamp_min(1e-6)
        child_rotations_unnorm = (
            rearrange(base_rotations, "b n k xyzw -> b (n k) xyzw") + delta_rotations
        )
        child_rotations = child_rotations_unnorm / (
            child_rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8
        )
        child_opacities_raw = rearrange(
            base_opacities_raw,
            "b n k one -> b (n k) one",
        ) + delta_opacities
        child_harmonics = rearrange(
            base_harmonics,
            "b n k c d_sh -> b (n k) c d_sh",
        ) + rearrange(delta_shs, "b nk (c d_sh) -> b nk c d_sh", c=3)
        child_covariances = build_covariance(child_scales, child_rotations)

        child_gaussians = Gaussians(
            child_means,
            child_covariances,
            child_rotations,
            child_scales,
            child_harmonics,
            child_opacities_raw.squeeze(-1).sigmoid(),
        )

        return {
            "gaussians": child_gaussians,
            "child_multiview_weights": child_multiview_features["weights"],
            "delta_scales": delta_scales,
            "delta_rotations": delta_rotations,
            "delta_opacities": delta_opacities,
            "delta_harmonics": delta_shs,
            "base_scales": rearrange(base_scales, "b n k xyz -> b (n k) xyz"),
            "base_opacities_raw": rearrange(base_opacities_raw, "b n k one -> b (n k) one"),
        }

    def forward(
            self,
            context: dict,
            global_step: int = 0,
            visualization_dump: Optional[dict] = None,
            target: Optional[dict] = None,
            warmup_pts3d: bool = False,
    ):
        context_image = context.get("image_lr", context["image"])
        target_image = target.get("image_lr", target["image"]) if target is not None else None
        context_image_sr = self._super_resolve(context_image) if self.swinir is not None else None

        b, v_cxt, _, h, w = context_image.shape

        if target is not None:
            v_tgt = target_image.shape[1]
            context_target = {
                "image": normalize_image(torch.cat([context_image, target_image], dim=1)),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1),
            }
            # Encode the context and target images.
            out = self.backbone(context_target, target_num_views=v_tgt)
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context_image),
                "intrinsics": context["intrinsics"],
            }
            # Encode the context images.
            out = self.backbone(context_input)

        dec_feat, shape, images = out['dec_feat'], out['shape'], out['images']
        resunet_features = None
        if self.resunet_token_fusion is not None:
            if context_image_sr is None:
                raise RuntimeError("use_resunet_token_fusion=True requires use_swinir=True")
            resunet_features = self.resunet_token_fusion(
                context_image_sr,
                dec_feat[-1][:, :v_cxt].float(),
            )

        with torch.amp.autocast('cuda', enabled=False):
            all_other_params = []
            all_depth_res = []

            if self.cfg.estimating_pose:
                all_pose_params = []

            if self.cfg.estimating_focal:
                all_intrin_params = []

            res1 = self._downstream_depth_head(1, [tok[:, 0].float() for tok in dec_feat], shape[:, 0])
            all_depth_res.append(res1)
            for i in range(1, v_cxt):
                res2 = self._downstream_depth_head(2, [tok[:, i].float() for tok in dec_feat], shape[:, i])
                all_depth_res.append(res2)

            # for the 3DGS heads
            if 'dpt' in self.gs_params_head_type:
                GS_res1 = self.gaussian_param_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3],
                                                   shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v_cxt):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3],
                                                        shape[0, i].cpu().tolist())
                    GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                    all_other_params.append(GS_res2)
            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

            # for pose head
            if self.cfg.estimating_pose:
                pose_feat = dec_feat if 'pose_feat' not in out else out['pose_feat']
                pose_res1 = self.pose_head([tok[:, 0].float() for tok in pose_feat],
                                           shape[0, 0].cpu().tolist())  # (16, 9)
                # print("pose_res1", pose_res1.keys())
                all_pose_params.append(pose_res1['pose'])
                if self.cfg.estimating_focal:
                    all_intrin_params.append(pose_res1['intrinsics'])
                for i in range(1, v_cxt + v_tgt):
                    pose_res2 = self.pose_head2([tok[:, i].float() for tok in pose_feat],
                                                shape[0, i].cpu().tolist())  # (16, 9)
                    all_pose_params.append(pose_res2['pose'])
                    if self.cfg.estimating_focal:
                        all_intrin_params.append(pose_res2['intrinsics'])

        gaussians = torch.stack(all_other_params, dim=1)  # [b, v, 65536, 83]
        # print("gaussians", gaussians.shape)

        if self.cfg.estimating_pose:
            poses_enc = torch.stack(all_pose_params, dim=1)  # (b, v 9)
            pred_extrinsics = self.process_pose(poses_enc, v_cxt)  # (b, v, 4, 4)
            # print("translation", pred_extrinsics[0,:,:3,-1])

        if self.cfg.estimating_focal:
            intrin_enc = torch.stack(all_intrin_params, dim=1)
            pred_intrinsics = self.process_intrinsics(intrin_enc, h, w)

        depth_all = [all_depth_res_i['depth'] for all_depth_res_i in all_depth_res]
        depth_all = torch.stack(depth_all, dim=1).squeeze(-1)  # [b, v, h, w]
        depths_per_view = depth_all

        context_extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        context_intrinsics = pred_intrinsics[:, :v_cxt] if self.cfg.estimating_focal else context["intrinsics"]

        point_map_from_depth = unproject_depth_map_to_point_map_batch(rearrange(depth_all, "b v ... -> (b v) ..."),
                                                                      rearrange(context_extrinsics,
                                                                                "b v ... -> (b v ) ..."),
                                                                      rearrange(context_intrinsics,
                                                                                "b v ... -> (b v ) ..."))

        depth_to_pts_all = rearrange(point_map_from_depth, "(b v) ... -> b v ...", b=b, v=v_cxt)
        depth_to_pts_all = rearrange(depth_to_pts_all, "b v h w xyz -> b v (h w) xyz")
        # print("depth_to_pts_all", depth_to_pts_all[0,0,0])
        flattened_anchors = rearrange(depth_to_pts_all, "b v r xyz -> b (v r) xyz")
        source_view_indices = repeat(
            torch.arange(v_cxt, device=depth_to_pts_all.device),
            "v -> b (v r)",
            b=b,
            r=depth_to_pts_all.shape[2],
        )
        image_only_consistency = None
        if self.image_only_consistency is not None:
            if resunet_features is None or "image_only_64" not in resunet_features:
                raise RuntimeError(
                    "use_image_only_feature_consistency=True requires "
                    "use_resunet_token_fusion=True and image_only_64 features."
                )
            image_only_consistency = self.image_only_consistency(
                flattened_anchors,
                resunet_features["image_only_64"],
                depth_all,
                context_extrinsics,
                context_intrinsics,
                source_view_indices,
            )
        anchor_multiview_features = None
        if self.cfg.use_anchor_multiview_feature_aggregation:
            if image_only_consistency is None:
                raise RuntimeError(
                    "use_anchor_multiview_feature_aggregation=True requires "
                    "use_image_only_feature_consistency=True."
                )
            if resunet_features is None or "256" not in resunet_features:
                raise RuntimeError(
                    "use_anchor_multiview_feature_aggregation=True requires "
                    "resunet_features['256']."
                )
            if context_image_sr is None:
                raise RuntimeError(
                    "use_anchor_multiview_feature_aggregation=True requires "
                    "use_swinir=True."
                )
            anchor_multiview_features = self._aggregate_anchor_multiview_features(
                flattened_anchors,
                resunet_features["256"],
                context_image_sr,
                image_only_consistency,
                context_extrinsics,
                context_intrinsics,
                source_view_indices,
            )
        depth_to_pts_all = depth_to_pts_all.unsqueeze(-2)
        gaussian_params = rearrange(gaussians, "... (srf c) -> ... srf c",
                                    srf=self.cfg.num_surfaces)  # for cfg.num_surfaces

        densities = gaussian_params[..., 0].sigmoid().unsqueeze(-1)

        gaussians = self.gaussian_adapter.forward(
            depth_to_pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_params[..., 1:], "b v r srf c -> b v r srf () c"),
        )
        point_offset_densification = None
        if self.point_offset_decoder is not None:
            if anchor_multiview_features is None:
                raise RuntimeError(
                    "use_point_offset_decoder=True requires "
                    "use_anchor_multiview_feature_aggregation=True."
                )
            point_offset_radius = rearrange(
                gaussians.scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ).detach().mean(dim=-1, keepdim=True)
            point_offset_densification = self.point_offset_decoder(
                flattened_anchors,
                anchor_multiview_features["feature_3d"],
                point_offset_radius,
            )
        child_gaussian_residual = None
        if self.child_gaussian_head is not None:
            if point_offset_densification is None:
                raise RuntimeError(
                    "use_child_gaussian_residual=True requires "
                    "use_point_offset_decoder=True."
                )
            if anchor_multiview_features is None:
                raise RuntimeError(
                    "use_child_gaussian_residual=True requires "
                    "use_anchor_multiview_feature_aggregation=True."
                )
            if resunet_features is None:
                raise RuntimeError(
                    "use_child_gaussian_residual=True requires "
                    "use_resunet_token_fusion=True."
                )
            if "256" not in resunet_features:
                raise RuntimeError(
                    "use_child_gaussian_residual=True requires "
                    "resunet_features['256']."
                )
            if context_image_sr is None:
                raise RuntimeError(
                    "use_child_gaussian_residual=True requires use_swinir=True."
                )
            child_gaussian_residual = self._decode_child_gaussian_residuals(
                gaussians,
                point_offset_densification,
                anchor_multiview_features,
                resunet_features["256"],
                context_image_sr,
                context_extrinsics,
                context_intrinsics,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view

            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )  # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )  # (b, v, h, w, 1, 1)

        encoder_output = dict()
        if context_image_sr is not None:
            encoder_output["context_image_sr"] = context_image_sr
        if resunet_features is not None:
            encoder_output["resunet_features"] = resunet_features
        if image_only_consistency is not None:
            source_view_indices = repeat(
                torch.arange(v_cxt, device=depth_all.device),
                "v -> b (v r)",
                b=b,
                r=h * w,
            )
            source_view_mask = F.one_hot(
                source_view_indices,
                num_classes=v_cxt,
            ).bool()
            cross_view_valid = (
                image_only_consistency.view_projection_valid
                & ~source_view_mask
            )
            score = (
                image_only_consistency.per_view_consistency_weight
                * cross_view_valid
            ).sum(dim=(1, 2)) / cross_view_valid.sum(dim=(1, 2)).clamp_min(1)
            encoder_output["image_only_feature_consistency"] = {
                "score": score,
                "inconsistency_score": 1 - score,
                "per_view_score": image_only_consistency.per_view_consistency_weight,
                "valid_mask": image_only_consistency.view_projection_valid,
                "cross_view_valid_mask": cross_view_valid,
                "occlusion_delta": image_only_consistency.occlusion_delta,
                "occlusion_valid_mask": image_only_consistency.occlusion_valid_mask,
            }
        if anchor_multiview_features is not None:
            encoder_output["anchor_multiview_features"] = anchor_multiview_features
        if point_offset_densification is not None:
            encoder_output["point_offset_densification"] = point_offset_densification
        if child_gaussian_residual is not None:
            encoder_output["child_gaussian_residual"] = child_gaussian_residual
            encoder_output["child_gaussians"] = child_gaussian_residual["gaussians"]

        encoder_output["gaussians"] = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = dict()
            encoder_output['extrinsics']['c'] = pred_extrinsics[:, :v_cxt]
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics

        if self.cfg.estimating_focal:
            encoder_output['intrinsics'] = dict()
            encoder_output['intrinsics']['c'] = pred_intrinsics[:, :v_cxt]
            if target is not None:
                encoder_output['intrinsics']['cwt'] = pred_intrinsics

        return encoder_output

    def process_pose(self, pose_enc, context_views):
        # pose_enc: (b v 9)
        b, v = pose_enc.shape[:2]
        poses = convert_pose_to_4x4(rearrange(pose_enc, "b v ... -> (b v) ..."))
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)

        if self.cfg.pose_make_baseline_1:
            a = poses[:, 0, :3, 3]  # [b, 3]
            b = poses[:, context_views - 1, :3, 3]  # [b, 3]
            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]
            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:, 0]  # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses  # [b,1,4,4] @ [b,v,4,4]

        return poses

    def process_intrinsics(self, intrin_enc, height, width):
        # intrin_enc: (b, v, 2)
        c_x = 0.5
        c_y = 0.5

        fov_h = intrin_enc[..., 0]
        if self.cfg.equal_fxfy:
            fov_w = fov_h
        else:
            fov_w = intrin_enc[..., 1]

        if self.cfg.equal_view_intrinsics:
            fov_h = fov_h[:, 0:1].repeat(1, intrin_enc.shape[1])
            fov_w = fov_w[:, 0:1].repeat(1, intrin_enc.shape[1])

        f_y = (height / 2.0) / torch.tan(fov_h / 2.0)
        f_x = (width / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(*intrin_enc.shape[:-1], 3, 3).to(intrin_enc.device)
        intrinsics[..., 0, 0] = f_x / width
        intrinsics[..., 1, 1] = f_y / height
        intrinsics[..., 0, 2] = c_x
        intrinsics[..., 1, 2] = c_y
        intrinsics[..., 2, 2] = 1.0

        return intrinsics

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
