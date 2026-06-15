from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.types import BatchedExample, DataShim
from ..super_resolution import FrozenSwinIRUpsampler
from ..types import Gaussians
from .resunet_fusion import HiSplatResUnetTokenFusion
from .multi_view_consistency import FeatureMultiViewConsistencyEstimator
from .gaussian_transmittance import GaussianTransmittanceEstimator
from .anchor_multiview_fusion import AnchorMultiViewFeatureFusion
from .backbone import BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import convert_pose_to_4x4, unproject_depth_map_to_point_map_batch
from .heads.pose_head import PoseHeadCfg
from ...global_cfg import get_cfg

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
    unimatch_weights_path: str = (
        "/space0/mengxl/NAS3R-master/pretrained_weights/"
        "gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
    )
    use_multiview_consistency: bool = True
    consistency_patch_size: int = 1
    consistency_depth_relative_tolerance: float = 0.1
    use_gaussian_transmittance: bool = True
    gaussian_transmittance_chunk_size: int = 256
    gaussian_transmittance_min_pixel_std: float = 0.3
    use_anchor_multiview_fusion: bool = True
    anchor_fusion_feature_dim: int = 128
    anchor_fusion_geometry_dim: int = 128
    anchor_fusion_geometry_num_frequencies: int = 6
    anchor_fusion_hidden_dim: int = 128
    anchor_fusion_occlusion_tau: float = 10.0
    anchor_num_slots: int = 8
    anchor_slot_dim: int = 32
    use_anchor_feature_unet: bool = True
    anchor_unet_base_dim: int = 128
    anchor_unet_attention_blocks: int = 4
    anchor_unet_num_heads: int = 4
    anchor_offset_hidden_dim: int = 256
    anchor_hr_feature_dim: int = 32
    depth_activation: str = 'sigmoid'

    equal_fxfy: bool = True
    equal_view_intrinsics: bool = True


def flatten_gaussians(gaussians) -> Gaussians:
    return Gaussians(
        rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
        rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
        rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
        rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
        rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
        rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
    )


def gather_gaussians(gaussians: Gaussians, indices: Tensor) -> Gaussians:
    def gather(values: Tensor) -> Tensor:
        index = indices
        for _ in range(values.ndim - 2):
            index = index.unsqueeze(-1)
        return values.gather(
            1,
            index.expand(-1, -1, *values.shape[2:]),
        )

    return Gaussians(
        means=gather(gaussians.means),
        covariances=gather(gaussians.covariances),
        rotations=gather(gaussians.rotations),
        scales=gather(gaussians.scales),
        harmonics=gather(gaussians.harmonics),
        opacities=gather(gaussians.opacities),
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
        if (
            self.resunet_token_fusion is not None
            and get_cfg().mode == "train"
            and cfg.unimatch_weights_path
        ):
            self.resunet_token_fusion.load_unimatch_encoder(
                cfg.unimatch_weights_path
            )
        self.multiview_consistency = (
            FeatureMultiViewConsistencyEstimator(
                patch_size=cfg.consistency_patch_size,
                depth_relative_tolerance=(
                    cfg.consistency_depth_relative_tolerance
                ),
            )
            if cfg.use_multiview_consistency
            else None
        )
        self.gaussian_transmittance = (
            GaussianTransmittanceEstimator(
                target_chunk_size=cfg.gaussian_transmittance_chunk_size,
                min_pixel_std=cfg.gaussian_transmittance_min_pixel_std,
            )
            if cfg.use_gaussian_transmittance
            else None
        )
        self.anchor_multiview_fusion = (
            AnchorMultiViewFeatureFusion(
                feature_dim=cfg.anchor_fusion_feature_dim,
                geometry_dim=cfg.anchor_fusion_geometry_dim,
                geometry_num_frequencies=(
                    cfg.anchor_fusion_geometry_num_frequencies
                ),
                hidden_dim=cfg.anchor_fusion_hidden_dim,
                occlusion_tau=cfg.anchor_fusion_occlusion_tau,
                num_slots=cfg.anchor_num_slots,
                slot_dim=cfg.anchor_slot_dim,
                use_feature_unet=cfg.use_anchor_feature_unet,
                unet_base_dim=cfg.anchor_unet_base_dim,
                unet_attention_blocks=cfg.anchor_unet_attention_blocks,
                unet_num_heads=cfg.anchor_unet_num_heads,
                offset_hidden_dim=cfg.anchor_offset_hidden_dim,
                hr_feature_dim=cfg.anchor_hr_feature_dim,
                raw_gaussian_dim=self.gaussian_adapter.d_in,
            )
            if cfg.use_anchor_multiview_fusion
            else None
        )
        self.child_gaussian_adapter = UnifiedGaussianAdapter(
            cfg.gaussian_adapter
        )

    def densify_prepared(
        self,
        encoder_output: dict,
        parent_selection: Tensor,
        global_step: int,
    ) -> dict:
        state = encoder_output.pop("_densification_state")
        (
            lr_gaussians,
            anchors,
            source_view_indices,
            fusion_features,
            depth_all,
            context_extrinsics,
            context_intrinsics,
            context_image_shape,
            anchor_grid_shape,
        ) = state
        b, num_parents = anchors.shape[:2]
        parent_selection_flat = parent_selection.reshape(b, num_parents)

        multiview_consistency = self.multiview_consistency(
            anchors=anchors,
            consistency_features=fusion_features["image_only_256"],
            depths=depth_all,
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            source_view_indices=source_view_indices,
        )
        gaussian_transmittance = self.gaussian_transmittance(
            means=anchors,
            covariances=rearrange(
                lr_gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            opacities=rearrange(
                lr_gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            image_shape=context_image_shape,
        )
        fused = self.anchor_multiview_fusion(
            anchors=anchors,
            feature_map=fusion_features["64"],
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            source_view_indices=source_view_indices,
            occlusion_delta=multiview_consistency.occlusion_delta,
            occlusion_valid_mask=multiview_consistency.occlusion_valid_mask,
            transmittance=gaussian_transmittance.transmittance,
            transmittance_valid_mask=gaussian_transmittance.valid_mask,
            consistency_weight=(
                multiview_consistency.per_view_consistency_weight
            ),
            consistency_valid_mask=(
                multiview_consistency.view_projection_valid
            ),
            anchor_grid_shape=anchor_grid_shape,
            hr_feature_map=fusion_features["256"],
            parent_selection=parent_selection_flat,
        )
        decoded = self.child_gaussian_adapter(
            means=fused.child_means,
            opacities=self.map_pdf_to_opacity(fused.densities, global_step),
            raw_gaussians=fused.raw_gaussians,
        )
        selected_sr = Gaussians(
            means=rearrange(decoded.means, "b n k xyz -> b (n k) xyz"),
            covariances=rearrange(
                decoded.covariances,
                "b n k i j -> b (n k) i j",
            ),
            rotations=rearrange(decoded.rotations, "b n k q -> b (n k) q"),
            scales=rearrange(decoded.scales, "b n k xyz -> b (n k) xyz"),
            harmonics=rearrange(
                decoded.harmonics,
                "b n k rgb sh -> b (n k) rgb sh",
            ),
            opacities=rearrange(decoded.opacities, "b n k -> b (n k)"),
        )

        lr_flat = flatten_gaussians(lr_gaussians)
        unselected_indices = torch.arange(
            num_parents,
            device=anchors.device,
        )[None].expand(b, -1)[~parent_selection_flat].reshape(b, -1)
        unselected_lr = gather_gaussians(lr_flat, unselected_indices)
        encoder_output["gaussians"] = Gaussians(
            means=torch.cat([unselected_lr.means, selected_sr.means], dim=1),
            covariances=torch.cat(
                [unselected_lr.covariances, selected_sr.covariances],
                dim=1,
            ),
            rotations=torch.cat(
                [unselected_lr.rotations, selected_sr.rotations],
                dim=1,
            ),
            scales=torch.cat([unselected_lr.scales, selected_sr.scales], dim=1),
            harmonics=torch.cat(
                [unselected_lr.harmonics, selected_sr.harmonics],
                dim=1,
            ),
            opacities=torch.cat(
                [unselected_lr.opacities, selected_sr.opacities],
                dim=1,
            ),
        )

        num_slots = decoded.means.shape[2]
        full_child_means = decoded.means.new_zeros(
            b,
            num_parents,
            num_slots,
            3,
        )
        full_child_means[parent_selection_flat] = decoded.means.reshape(
            -1,
            num_slots,
            3,
        )
        v, h, w, srf, spp = anchor_grid_shape
        encoder_output["child_means_for_grid_loss"] = rearrange(
            full_child_means,
            "b (v h w srf spp) k xyz -> b v h w srf spp k xyz",
            v=v,
            h=h,
            w=w,
            srf=srf,
            spp=spp,
        )
        return encoder_output

    def densify_from_render_error(
        self,
        encoder_output: dict,
        decoder,
        context_image: Tensor,
        context_extrinsics: Tensor,
        context_intrinsics: Tensor,
        context_near: Tensor,
        context_far: Tensor,
        global_step: int,
        budget_per_view: int,
        temperature: float,
        stochastic: bool,
    ) -> tuple[dict, Tensor, Tensor]:
        if temperature <= 0:
            raise ValueError("Densification temperature must be positive.")

        gaussians_lr = encoder_output["gaussians_lr"]
        v, h, w, srf, spp = encoder_output["densification_grid_shape"]
        with torch.no_grad():
            lr_render = decoder.forward(
                gaussians_lr,
                context_extrinsics,
                context_intrinsics,
                context_near,
                context_far,
                context_image.shape[-2:],
                depth_mode=None,
            )
            error_256 = (
                lr_render.color - context_image
            ).abs().mean(dim=2)
            error_64 = torch.nn.functional.adaptive_avg_pool2d(
                rearrange(error_256, "b v h w -> (b v) 1 h w"),
                (h, w),
            )
            error_64 = rearrange(
                error_64,
                "(b v) 1 h w -> b v h w",
                b=context_image.shape[0],
                v=v,
            )

            parents_per_view = h * w * srf * spp
            parent_score = error_64[..., None, None].expand(
                -1,
                -1,
                -1,
                -1,
                srf,
                spp,
            ).reshape(context_image.shape[0], v, parents_per_view)
            probability = torch.softmax(
                parent_score / temperature,
                dim=-1,
            )
            budget = min(budget_per_view, parents_per_view)
            if stochastic and budget < parents_per_view:
                selected_local = torch.multinomial(
                    probability.reshape(-1, parents_per_view),
                    budget,
                    replacement=False,
                ).reshape(context_image.shape[0], v, budget)
            else:
                selected_local = probability.topk(budget, dim=-1).indices

            view_offsets = (
                torch.arange(v, device=probability.device)
                * parents_per_view
            )[None, :, None]
            selected_global = (
                selected_local + view_offsets
            ).reshape(context_image.shape[0], v * budget)
            parent_selection = torch.zeros(
                context_image.shape[0],
                v * parents_per_view,
                dtype=torch.bool,
                device=probability.device,
            )
            parent_selection.scatter_(1, selected_global, True)
            parent_selection = parent_selection.reshape(
                context_image.shape[0],
                v,
                h,
                w,
                srf,
                spp,
            )

        encoder_output = self.densify_prepared(
            encoder_output,
            parent_selection,
            global_step,
        )
        return encoder_output, parent_selection, error_64.detach()

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
            pdf: Tensor,
            global_step: int,
    ) -> Tensor:
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
            prepare_densification: bool = False,
    ):
        context_image = context.get("image_lr", context["image"])
        target_image = target.get("image_lr", target["image"]) if target is not None else None
        context_image_sr = (
            self.swinir_upsampler(context_image)
            if self.swinir_upsampler is not None
            else context["image"]
        )

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
        lr_gaussians = gaussians
        (
            _,
            num_lr_views,
            num_lr_rays,
            num_lr_surfaces,
            num_lr_samples_per_pixel,
            _,
        ) = lr_gaussians.means.shape
        anchors = rearrange(
            lr_gaussians.means,
            "b v r srf spp xyz -> b (v r srf spp) xyz",
        )
        anchors_per_view = (
            num_lr_rays * num_lr_surfaces * num_lr_samples_per_pixel
        )
        source_view_indices = torch.arange(
            num_lr_views,
            device=anchors.device,
        ).view(1, num_lr_views, 1).expand(
            b,
            num_lr_views,
            anchors_per_view,
        )
        source_view_indices = rearrange(
            source_view_indices,
            "b v n -> b (v n)",
        )

        if prepare_densification:
            encoder_output = {
                "gaussians_lr": flatten_gaussians(lr_gaussians),
                "densification_grid_shape": (
                    num_lr_views,
                    h,
                    w,
                    num_lr_surfaces,
                    num_lr_samples_per_pixel,
                ),
                "_densification_state": (
                    lr_gaussians,
                    anchors,
                    source_view_indices,
                    fusion_features,
                    depth_all,
                    context_extrinsics,
                    context_intrinsics,
                    context_image_sr.shape[-2:],
                    (
                        num_lr_views,
                        h,
                        w,
                        num_lr_surfaces,
                        num_lr_samples_per_pixel,
                    ),
                ),
            }
            if self.cfg.estimating_pose:
                encoder_output["extrinsics"] = {
                    "c": pred_extrinsics[:, :v_cxt],
                }
                if target is not None:
                    encoder_output["extrinsics"]["cwt"] = pred_extrinsics
            if self.cfg.estimating_focal:
                encoder_output["intrinsics"] = {
                    "c": pred_intrinsics[:, :v_cxt],
                }
                if target is not None:
                    encoder_output["intrinsics"]["cwt"] = pred_intrinsics
            if visualization_dump is not None:
                visualization_dump["depth"] = depths_per_view
                visualization_dump["means"] = rearrange(
                    lr_gaussians.means,
                    "b v (h w) srf spp xyz -> b v h w (srf spp) xyz",
                    h=h,
                    w=w,
                )
            return encoder_output

        multiview_consistency = None
        if self.multiview_consistency is not None:
            if fusion_features is None:
                raise RuntimeError(
                    "Multi-view consistency requires use_resunet_fusion=True."
                )
            multiview_consistency = self.multiview_consistency(
                anchors=anchors,
                consistency_features=fusion_features["image_only_256"],
                depths=depth_all,
                extrinsics=context_extrinsics,
                intrinsics=context_intrinsics,
                source_view_indices=source_view_indices,
            )

        gaussian_transmittance = None
        if self.gaussian_transmittance is not None:
            gaussian_transmittance = self.gaussian_transmittance(
                means=anchors,
                covariances=rearrange(
                    lr_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                opacities=rearrange(
                    lr_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                extrinsics=context_extrinsics,
                intrinsics=context_intrinsics,
                image_shape=context_image_sr.shape[-2:],
            )
        anchor_multiview_fusion = None
        decoded_gaussians = None
        if self.anchor_multiview_fusion is not None:
            if fusion_features is None:
                raise RuntimeError(
                    "Anchor multi-view fusion requires use_resunet_fusion=True."
                )
            if multiview_consistency is None:
                raise RuntimeError(
                    "Anchor multi-view fusion requires "
                    "use_multiview_consistency=True."
                )
            if gaussian_transmittance is None:
                raise RuntimeError(
                    "Anchor multi-view fusion requires "
                    "use_gaussian_transmittance=True."
                )
            anchor_multiview_fusion = self.anchor_multiview_fusion(
                anchors=anchors,
                feature_map=fusion_features["64"],
                extrinsics=context_extrinsics,
                intrinsics=context_intrinsics,
                source_view_indices=source_view_indices,
                occlusion_delta=multiview_consistency.occlusion_delta,
                occlusion_valid_mask=(
                    multiview_consistency.occlusion_valid_mask
                ),
                transmittance=gaussian_transmittance.transmittance,
                transmittance_valid_mask=(
                    gaussian_transmittance.valid_mask
                ),
                consistency_weight=(
                    multiview_consistency.per_view_consistency_weight
                ),
                consistency_valid_mask=(
                    multiview_consistency.view_projection_valid
                ),
                anchor_grid_shape=(
                    num_lr_views,
                    h,
                    w,
                    num_lr_surfaces,
                    num_lr_samples_per_pixel,
                ),
                hr_feature_map=fusion_features["256"],
            )
            decoded = self.child_gaussian_adapter(
                means=anchor_multiview_fusion.child_means,
                opacities=self.map_pdf_to_opacity(
                    anchor_multiview_fusion.densities,
                    global_step,
                ),
                raw_gaussians=anchor_multiview_fusion.raw_gaussians,
            )
            decoded_gaussians = Gaussians(
                means=rearrange(
                    decoded.means,
                    "b n k xyz -> b (n k) xyz",
                ),
                covariances=rearrange(
                    decoded.covariances,
                    "b n k i j -> b (n k) i j",
                ),
                rotations=rearrange(
                    decoded.rotations,
                    "b n k q -> b (n k) q",
                ),
                scales=rearrange(
                    decoded.scales,
                    "b n k xyz -> b (n k) xyz",
                ),
                harmonics=rearrange(
                    decoded.harmonics,
                    "b n k rgb sh -> b (n k) rgb sh",
                ),
                opacities=rearrange(
                    decoded.opacities,
                    "b n k -> b (n k)",
                ),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view
            visualization_dump["scales"] = rearrange(
                lr_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                lr_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                lr_gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )  # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                lr_gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )  # (b, v, h, w, 1, 1)

        encoder_output = dict()
        encoder_output["gaussians_lr"] = flatten_gaussians(lr_gaussians)
        encoder_output["gaussians"] = (
            decoded_gaussians
            if decoded_gaussians is not None
            else flatten_gaussians(lr_gaussians)
        )
        if anchor_multiview_fusion is not None:
            encoder_output["child_means_for_grid_loss"] = rearrange(
                decoded.means,
                "b (v h w srf spp) k xyz -> b v h w srf spp k xyz",
                v=num_lr_views,
                h=h,
                w=w,
                srf=num_lr_surfaces,
                spp=num_lr_samples_per_pixel,
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
