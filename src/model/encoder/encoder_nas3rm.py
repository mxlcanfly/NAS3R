from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.types import BatchedExample, DataShim
from ..super_resolution import FrozenSwinIRUpsampler
from ..types import Gaussians
from .anchor_feature_sampler import AnchorFeatureAggregator, AnchorFeatureSampler
from .anchor_geometry_encoder import AnchorGeometryEncoder
from .anchor_gaussian_decoder import (
    AnchorGaussianResidualDecoder,
    scale_gaussian_scaffold,
)
from .anchor_litept_fusion import AnchorLitePTFusion
from .resunet_fusion import HiSplatResUnetTokenFusion
from .backbone import BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import convert_pose_to_4x4, unproject_depth_map_to_point_map_batch
from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam
from .heads.pose_head import PoseHeadCfg
from .local_entropy import compute_local_shannon_entropy

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

    use_swinir_sr: bool = True
    swinir_weight_path: str = "/space0/mengxl/NAS3R-master/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    use_resunet_fusion: bool = True
    use_anchor_feature_sampler: bool = False
    use_anchor_feature_aggregator: bool = False
    anchor_feature_patch_size: int = 4
    anchor_feature_max_anchors: int | None = None
    anchor_feature_patch_dim: int = 512
    anchor_feature_num_views: int = 2
    anchor_feature_view_dim: int = 128
    anchor_feature_out_dim: int = 256
    use_anchor_geometry_query: bool = False
    anchor_geometry_num_frequencies: int = 6
    anchor_geometry_knn: int = 8
    anchor_geometry_query_dim: int = 256
    anchor_geometry_hidden_dim: int = 512
    anchor_geometry_knn_chunk_size: int = 256
    use_anchor_litept_fusion: bool = False
    anchor_litept_path: str = "/space0/mengxl/LitePT-main"
    anchor_litept_use_full: bool = True
    anchor_litept_grid_size: float = 0.02
    anchor_litept_token_dim: int = 512
    anchor_litept_output_dim: int = 128
    anchor_litept_enc_channels: list[int] | None = None
    anchor_litept_enc_num_head: list[int] | None = None
    anchor_litept_dec_channels: list[int] | None = None
    anchor_litept_dec_num_head: list[int] | None = None
    anchor_litept_fallback_blocks: int = 2
    anchor_litept_fallback_knn: int = 16
    anchor_litept_fallback_chunk_size: int = 1024
    use_anchor_gaussian_decoder: bool = True
    anchor_gaussian_decoder_hidden_dim: int = 512
    anchor_gaussians_per_anchor: int = 8
    anchor_scaffold_scale_divisor: float = 2.4
    anchor_scaffold_opacity_multiplier: float = 0.6
    anchor_child_offset_radius: float = 0.75
    anchor_child_scale_residual_range: float = 0.5
    anchor_child_opacity_residual_range: float = 2.0
    anchor_child_rotation_residual_range: float = 0.1
    anchor_child_sh_residual_range: float = 0.1
    entropy_num_gray_levels: int = 256
    entropy_window_size: int = 9

    depth_activation: str = 'sigmoid'

    equal_fxfy: bool = True
    equal_view_intrinsics: bool = True


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


def flatten_gaussians(gaussians) -> Gaussians:
    return Gaussians(
        rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
        rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
        rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
        rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
        rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
        rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
    )


class EncoderNAS3RM(Encoder[EncoderNAS3RMCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNAS3RMCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

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

        self.swinir_upsampler = (
            FrozenSwinIRUpsampler(cfg.swinir_weight_path)
            if cfg.use_swinir_sr
            else None
        )
        self.resunet_token_fusion = (
            HiSplatResUnetTokenFusion(token_dim=self.backbone.dec_embed_dim)
            if cfg.use_resunet_fusion
            else None
        )
        self.anchor_feature_sampler = (
            AnchorFeatureSampler(
                patch_size=cfg.anchor_feature_patch_size,
            )
            if cfg.use_anchor_feature_sampler
            else None
        )
        self.anchor_feature_aggregator = (
            AnchorFeatureAggregator(
                patch_feature_dim=cfg.anchor_feature_patch_dim,
                num_views=cfg.anchor_feature_num_views,
                view_dim=cfg.anchor_feature_view_dim,
                out_dim=cfg.anchor_feature_out_dim,
            )
            if cfg.use_anchor_feature_aggregator
            else None
        )
        self.anchor_geometry_encoder = (
            AnchorGeometryEncoder(
                num_frequencies=cfg.anchor_geometry_num_frequencies,
                knn=cfg.anchor_geometry_knn,
                query_dim=cfg.anchor_geometry_query_dim,
                hidden_dim=cfg.anchor_geometry_hidden_dim,
                knn_chunk_size=cfg.anchor_geometry_knn_chunk_size,
            )
            if cfg.use_anchor_geometry_query
            else None
        )
        self.anchor_litept_fusion = (
            AnchorLitePTFusion(
                feature_dim=cfg.anchor_feature_out_dim,
                geometry_dim=cfg.anchor_geometry_query_dim,
                token_dim=cfg.anchor_litept_token_dim,
                litept_path=cfg.anchor_litept_path,
                use_full_litept=cfg.anchor_litept_use_full,
                litept_grid_size=cfg.anchor_litept_grid_size,
                litept_output_dim=cfg.anchor_litept_output_dim,
                litept_enc_channels=tuple(
                    cfg.anchor_litept_enc_channels
                    or [64, 128, 256, 384, 504]
                ),
                litept_enc_num_head=tuple(
                    cfg.anchor_litept_enc_num_head
                    or [4, 8, 8, 16, 28]
                ),
                litept_dec_channels=tuple(
                    cfg.anchor_litept_dec_channels
                    or [128, 128, 256, 384]
                ),
                litept_dec_num_head=tuple(
                    cfg.anchor_litept_dec_num_head
                    or [8, 8, 8, 12]
                ),
                fallback_blocks=cfg.anchor_litept_fallback_blocks,
                fallback_knn=cfg.anchor_litept_fallback_knn,
                fallback_chunk_size=cfg.anchor_litept_fallback_chunk_size,
            )
            if cfg.use_anchor_litept_fusion
            else None
        )
        self.anchor_gaussian_decoder = (
            AnchorGaussianResidualDecoder(
                token_dim=cfg.anchor_litept_token_dim,
                gaussians_per_anchor=cfg.anchor_gaussians_per_anchor,
                sh_degree=cfg.gaussian_adapter.sh_degree,
                offset_radius=cfg.anchor_child_offset_radius,
                scale_residual_range=cfg.anchor_child_scale_residual_range,
                opacity_residual_range=cfg.anchor_child_opacity_residual_range,
                rotation_residual_range=cfg.anchor_child_rotation_residual_range,
                sh_residual_range=cfg.anchor_child_sh_residual_range,
                hidden_dim=cfg.anchor_gaussian_decoder_hidden_dim,
            )
            if cfg.use_anchor_gaussian_decoder
            else None
        )

    def _sample_child_entropy_from_source_view(
        self,
        child_means: torch.Tensor,
        entropy_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_anchors, num_children, _ = child_means.shape
        if num_anchors % num_views != 0:
            raise ValueError(
                "Expected anchors to be grouped by source view, got "
                f"{num_anchors} anchors and {num_views} views."
            )
        anchors_per_view = num_anchors // num_views
        child_means = rearrange(
            child_means,
            "b (v r) k xyz -> b v (r k) xyz",
            v=num_views,
            r=anchors_per_view,
        )

        cam_points = transform_world2cam(
            homogenize_points(child_means),
            extrinsics[:, :, None],
        )[..., :-1]
        camera_depth = cam_points[..., -1]
        projected_xy = project_camera_space(cam_points, intrinsics[:, :, None])
        valid = (
            (camera_depth > 1e-6)
            & (projected_xy[..., 0] >= 0)
            & (projected_xy[..., 0] <= 1)
            & (projected_xy[..., 1] >= 0)
            & (projected_xy[..., 1] <= 1)
        )

        sampled = F.grid_sample(
            rearrange(entropy_map, "b v h w -> (b v) 1 h w"),
            rearrange(projected_xy * 2 - 1, "b v m xy -> (b v) m () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = rearrange(
            sampled.squeeze(1).squeeze(-1),
            "(b v) (r k) -> b (v r) k",
            b=b,
            v=num_views,
            r=anchors_per_view,
            k=num_children,
        )
        valid = rearrange(
            valid,
            "b v (r k) -> b (v r) k",
            r=anchors_per_view,
            k=num_children,
        )
        return sampled, valid

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
        context_image_sr = (
            self.swinir_upsampler(context_image)
            if self.swinir_upsampler is not None
            else context["image"]
        )
        sr_entropy = compute_local_shannon_entropy(
            context_image_sr,
            num_gray_levels=self.cfg.entropy_num_gray_levels,
            window_size=self.cfg.entropy_window_size,
        )

        device = context_image.device
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
        fusion_features = None
        if self.resunet_token_fusion is not None:
            fusion_features = self.resunet_token_fusion(
                context_image_sr[:, :v_cxt],
                dec_feat[-1][:, :v_cxt],
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
        depth_to_pts_all = depth_to_pts_all.unsqueeze(-2)
        gaussian_params = rearrange(gaussians, "... (srf c) -> ... srf c",
                                    srf=self.cfg.num_surfaces)  # for cfg.num_surfaces

        densities = gaussian_params[..., 0].sigmoid().unsqueeze(-1)

        gaussians = self.gaussian_adapter.forward(
            depth_to_pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_params[..., 1:], "b v r srf c -> b v r srf () c"),
        )
        anchor_feature_samples = None
        anchor_features = None
        anchor_litept_output = None
        anchor_child_output = None
        anchor_geometry_query = None
        anchor_spacing = None
        anchors = None
        if self.anchor_feature_sampler is not None or self.anchor_geometry_encoder is not None:
            anchors = rearrange(depth_to_pts_all.squeeze(-2), "b v r xyz -> b (v r) xyz")
            if self.cfg.anchor_feature_max_anchors is not None:
                anchors = anchors[:, :self.cfg.anchor_feature_max_anchors]

        if self.anchor_feature_sampler is not None:
            if fusion_features is None or "256" not in fusion_features:
                raise RuntimeError("Anchor feature sampling requires use_resunet_fusion=True.")

            anchor_feature_samples = self.anchor_feature_sampler(
                anchors,
                fusion_features["256"],
                context_extrinsics,
                context_intrinsics,
            )
            if self.anchor_feature_aggregator is not None:
                anchor_features = self.anchor_feature_aggregator(anchor_feature_samples)
        if self.anchor_geometry_encoder is not None:
            geometry_encoding = self.anchor_geometry_encoder(anchors)
            anchor_geometry_query = geometry_encoding.query
            anchor_spacing = geometry_encoding.spacing
        if self.anchor_litept_fusion is not None:
            if anchor_features is None:
                raise RuntimeError(
                    "Anchor LitePT fusion requires use_anchor_feature_aggregator=True."
                )
            if anchor_geometry_query is None:
                raise RuntimeError(
                    "Anchor LitePT fusion requires use_anchor_geometry_query=True."
                )
            anchor_litept_output = self.anchor_litept_fusion(
                anchors,
                anchor_features,
                anchor_geometry_query,
            )
        scaffold_gaussians = scale_gaussian_scaffold(
            flatten_gaussians(gaussians),
            scale_divisor=self.cfg.anchor_scaffold_scale_divisor,
            opacity_multiplier=self.cfg.anchor_scaffold_opacity_multiplier,
        )
        final_gaussians = scaffold_gaussians
        if self.anchor_gaussian_decoder is not None:
            if anchor_litept_output is None:
                raise RuntimeError(
                    "Anchor Gaussian decoder requires use_anchor_litept_fusion=True."
                )
            if anchors is None or anchor_spacing is None:
                raise RuntimeError(
                    "Anchor Gaussian decoder requires anchor geometry encoding."
                )
            if scaffold_gaussians.means.shape[1] != anchors.shape[1]:
                raise RuntimeError(
                    "Anchor Gaussian decoder expects one scaffold Gaussian per anchor, "
                    f"got {scaffold_gaussians.means.shape[1]} Gaussians and "
                    f"{anchors.shape[1]} anchors."
                )
            anchor_child_output = self.anchor_gaussian_decoder(
                anchor_litept_output,
                anchors,
                scaffold_gaussians,
                anchor_spacing,
            )
            final_gaussians = anchor_child_output["gaussians"]

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view
            if fusion_features is not None:
                visualization_dump["fusion_features"] = fusion_features

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
        encoder_output["sr_entropy"] = sr_entropy
        encoder_output["context_lr_depth"] = depths_per_view
        if fusion_features is not None:
            encoder_output["fusion_features"] = fusion_features
        if anchor_feature_samples is not None:
            encoder_output["anchor_feature_samples"] = anchor_feature_samples
        if anchor_features is not None:
            encoder_output["anchor_features"] = anchor_features
        if anchor_litept_output is not None:
            encoder_output["anchor_litept_features"] = anchor_litept_output
        if anchor_child_output is not None:
            encoder_output["anchor_child_offsets"] = anchor_child_output["offsets"]
            encoder_output["anchor_child_raw_opacity"] = anchor_child_output["raw_opacity"]
            encoder_output["anchor_child_opacities"] = anchor_child_output["opacities"]
        if anchors is not None:
            encoder_output["anchors"] = anchors
        if anchor_geometry_query is not None:
            encoder_output["anchor_geometry_query"] = anchor_geometry_query
            encoder_output["anchor_spacing"] = anchor_spacing

        encoder_output["gaussians"] = final_gaussians

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
