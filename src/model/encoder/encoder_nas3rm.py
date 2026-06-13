from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import camera_head_factory, head_factory
from ...dataset.shims.normalize_shim import (
    apply_normalize_shim,
    normalize_image,
)
from ...dataset.types import BatchedExample, DataShim
from ..super_resolution import FrozenSwinIRUpsampler
from ..types import Gaussians
from .backbone import BackboneCfg, get_backbone
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import (
    convert_pose_to_4x4,
    unproject_depth_map_to_point_map_batch,
)
from .heads.pose_head import PoseHeadCfg
from .anchor_feature_sampler import AnchorFeatureSampler
from .anchor_geometry_encoder import PointGeometryEncoder
from .anchor_litept_fusion import AnchorLitePTFusion
from .common.gaussians import build_covariance
from .entropy_guided_points import EntropyGuidedPointSampler
from .gaussian_residual_refiner import GaussianResidualRefiner
from .resunet_fusion import HiSplatResUnetTokenFusion

inf = float("inf")


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
    pose_head_type: str = "mlp"
    estimating_focal: bool = False
    estimating_pose: bool = True
    use_swinir_sr: bool = True
    swinir_weight_path: str = (
        "/space0/mengxl/NAS3R-master/pretrained_weights/"
        "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    )
    entropy_num_gray_levels: int = 256
    entropy_window_size: int = 9
    entropy_points_per_view: int = 4096
    multiview_patch_size: int = 2
    multiview_depth_tolerance: float = 0.05
    geometry_num_frequencies: int = 6
    geometry_dim: int = 128
    visual_feature_dim: int = 256
    intrinsic_feature_dim: int = 64
    point_type_dim: int = 16
    litept_token_dim: int = 512
    gaussian_refiner_hidden_dim: int = 512
    sr_initial_opacity: float = 0.05
    sr_initial_covariance: float = 0.02

    depth_activation: str = "sigmoid"

    equal_fxfy: bool = True
    equal_view_intrinsics: bool = True


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

        if self.cfg.depth_activation == "exp":
            self.set_depth_head(
                output_mode="depth",
                head_type="dpt",
                landscape_only=True,
                depth_mode=("exp", -inf, inf),
                conf_mode=None,
            )
        elif self.cfg.depth_activation == "sigmoid":
            self.set_depth_head(
                output_mode="depth",
                head_type="dpt",
                landscape_only=True,
                depth_mode=("range", 1, 100.0),
                conf_mode=None,
            )
        else:
            raise NotImplementedError

        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        if self.cfg.estimating_pose:
            self.set_pose_head(cfg, cfg.pose_head_type)

        self.swinir_upsampler = (
            FrozenSwinIRUpsampler(cfg.swinir_weight_path) if cfg.use_swinir_sr else None
        )
        self.resunet_token_fusion = HiSplatResUnetTokenFusion(
            token_dim=self.backbone.dec_embed_dim,
            build_condition_head=False,
        )
        self.lr_feature_sampler = AnchorFeatureSampler(
            patch_size=cfg.multiview_patch_size,
            padding_mode="border",
        )
        self.sr_feature_sampler = AnchorFeatureSampler(
            patch_size=cfg.multiview_patch_size,
            padding_mode="border",
        )
        self.entropy_point_sampler = EntropyGuidedPointSampler(
            points_per_view=cfg.entropy_points_per_view,
            num_gray_levels=cfg.entropy_num_gray_levels,
            window_size=cfg.entropy_window_size,
        )
        self.point_geometry_encoder = PointGeometryEncoder(
            num_frequencies=cfg.geometry_num_frequencies,
            output_dim=cfg.geometry_dim,
        )
        patch_area = cfg.multiview_patch_size**2
        self.lr_visual_proj = nn.Sequential(
            nn.Linear(patch_area * (128 + 3), cfg.visual_feature_dim),
            nn.LayerNorm(cfg.visual_feature_dim),
            nn.GELU(),
        )
        self.sr_visual_proj = nn.Sequential(
            nn.Linear(patch_area * (32 + 3), cfg.visual_feature_dim),
            nn.LayerNorm(cfg.visual_feature_dim),
            nn.GELU(),
        )
        # CroCo encodes each normalized 3x3 intrinsic matrix from its nine
        # entries. Use the same input representation with a compact pointwise
        # embedding instead of CroCo's 1024-dimensional view token.
        self.intrinsic_encoder = nn.Sequential(
            nn.Linear(9, cfg.intrinsic_feature_dim),
            nn.LayerNorm(cfg.intrinsic_feature_dim),
            nn.GELU(),
        )
        self.point_type_embedding = nn.Embedding(2, cfg.point_type_dim)
        gaussian_parameter_dim = (
            3
            + 4
            + 1
            + 3 * (cfg.gaussian_adapter.sh_degree + 1) ** 2
        )
        point_feature_dim = (
            cfg.visual_feature_dim
            + gaussian_parameter_dim
            + cfg.intrinsic_feature_dim
            + cfg.point_type_dim
        )
        self.anchor_litept_fusion = AnchorLitePTFusion(
            feature_dim=point_feature_dim,
            geometry_dim=cfg.geometry_dim,
            token_dim=cfg.litept_token_dim,
        )
        self.gaussian_residual_refiner = GaussianResidualRefiner(
            token_dim=cfg.litept_token_dim,
            sh_degree=cfg.gaussian_adapter.sh_degree,
            hidden_dim=cfg.gaussian_refiner_hidden_dim,
        )
        self._freeze_lr_network()

    def _freeze_lr_network(self) -> None:
        self._frozen_lr_modules = (
            self.backbone,
            self.downstream_depth_head1,
            self.downstream_depth_head2,
            self.gaussian_param_head,
            self.gaussian_param_head2,
        )
        if self.cfg.estimating_pose:
            self._frozen_lr_modules += (self.pose_head, self.pose_head2)
        for module in self._frozen_lr_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen_lr_modules:
            module.eval()
        return self

    @staticmethod
    def _concatenate_gaussians(
        first: Gaussians,
        second: Gaussians,
    ) -> Gaussians:
        return Gaussians(
            means=torch.cat((first.means, second.means), dim=1),
            covariances=torch.cat(
                (first.covariances, second.covariances),
                dim=1,
            ),
            rotations=torch.cat((first.rotations, second.rotations), dim=1),
            scales=torch.cat((first.scales, second.scales), dim=1),
            harmonics=torch.cat((first.harmonics, second.harmonics), dim=1),
            opacities=torch.cat((first.opacities, second.opacities), dim=1),
        )

    @staticmethod
    def _detach_gaussians(gaussians: Gaussians) -> Gaussians:
        return Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            rotations=gaussians.rotations.detach(),
            scales=gaussians.scales.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
        )

    @staticmethod
    def _slice_gaussians(gaussians: Gaussians, start: int, end: int) -> Gaussians:
        return Gaussians(
            means=gaussians.means[:, start:end],
            covariances=gaussians.covariances[:, start:end],
            rotations=gaussians.rotations[:, start:end],
            scales=gaussians.scales[:, start:end],
            harmonics=gaussians.harmonics[:, start:end],
            opacities=gaussians.opacities[:, start:end],
        )

    @staticmethod
    def _aggregate_multiview_features(samples) -> Tensor:
        mask = samples.valid_mask.to(dtype=samples.features.dtype)
        return (samples.features * mask[..., None]).sum(dim=2)

    @staticmethod
    def _pack_gaussian_parameters(gaussians: Gaussians) -> Tensor:
        return torch.cat(
            (
                gaussians.scales.detach(),
                gaussians.rotations.detach(),
                torch.logit(
                    gaussians.opacities.detach(),
                    eps=1e-6,
                )[..., None],
                gaussians.harmonics.detach().flatten(start_dim=-2),
            ),
            dim=-1,
        )

    @staticmethod
    def _source_view_indices(
        batch_size: int,
        num_views: int,
        points_per_view: int,
        device: torch.device,
    ) -> Tensor:
        source_view = torch.arange(device=device, end=num_views)
        source_view = source_view.repeat_interleave(points_per_view)
        return source_view[None].expand(batch_size, -1)

    @staticmethod
    def _gather_by_index(values: Tensor, index: Tensor) -> Tensor:
        gather_index = index
        while gather_index.ndim < values.ndim:
            gather_index = gather_index.unsqueeze(-1)
        gather_index = gather_index.expand(*index.shape, *values.shape[2:])
        return values.gather(dim=1, index=gather_index)

    def _encode_point_intrinsics(
        self,
        intrinsics: Tensor,
        lr_source_view: Tensor,
        sr_source_view: Tensor,
    ) -> Tensor:
        view_embeddings = self.intrinsic_encoder(intrinsics.flatten(start_dim=2))
        lr_embeddings = self._gather_by_index(
            view_embeddings,
            lr_source_view,
        )
        sr_embeddings = self._gather_by_index(
            view_embeddings,
            sr_source_view,
        )
        return torch.cat((lr_embeddings, sr_embeddings), dim=1)

    def _initialize_sr_gaussians(
        self,
        means: Tensor,
        source_view: Tensor,
        source_pixel_index: Tensor,
        sr_images: Tensor,
    ) -> Gaussians:
        b, _, _, sr_h, sr_w = sr_images.shape
        if source_view.shape != source_pixel_index.shape:
            raise ValueError(
                "source_view and source_pixel_index must have matching shapes."
            )
        if source_view.shape[0] != b:
            raise ValueError(
                "SR source indices and SR images must have matching batch sizes."
            )

        global_pixel_index = source_view * (sr_h * sr_w) + source_pixel_index
        sr_rgb = rearrange(
            sr_images,
            "b v c h w -> b (v h w) c",
        ).gather(
            dim=1,
            index=global_pixel_index[..., None].expand(-1, -1, 3),
        )

        harmonics = means.new_zeros(
            (*means.shape[:2], 3, self.gaussian_adapter.d_sh)
        )
        harmonics[..., 0] = (sr_rgb - 0.5) / 0.28209479177387814

        scale = self.cfg.sr_initial_covariance**0.5
        scales = means.new_full(means.shape, scale)
        rotations = means.new_zeros((*means.shape[:2], 4))
        rotations[..., 3] = 1
        opacities = means.new_full(
            means.shape[:2],
            self.cfg.sr_initial_opacity,
        )
        return Gaussians(
            means=means,
            covariances=build_covariance(scales, rotations),
            rotations=rotations,
            scales=scales,
            harmonics=harmonics,
            opacities=opacities,
        )

    def set_depth_head(
        self, output_mode, head_type, landscape_only, depth_mode, conf_mode
    ):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_depth_head1 = head_factory(
            head_type, output_mode, self.backbone, has_conf=bool(conf_mode)
        )
        self.downstream_depth_head2 = head_factory(
            head_type, output_mode, self.backbone, has_conf=bool(conf_mode)
        )

        # magic wrapper
        self.depth_head1 = transpose_to_landscape(
            self.downstream_depth_head1, activate=landscape_only
        )
        self.depth_head2 = transpose_to_landscape(
            self.downstream_depth_head2, activate=landscape_only
        )

    def set_gs_params_head(self, cfg, head_type):
        if head_type == "linear":
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size**2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)

        elif "dpt" in head_type:
            self.gaussian_param_head = head_factory(
                head_type,
                "gs_params",
                self.backbone,
                has_conf=False,
                out_nchan=self.raw_gs_dim,
            )
            self.gaussian_param_head2 = head_factory(
                head_type,
                "gs_params",
                self.backbone,
                has_conf=False,
                out_nchan=self.raw_gs_dim,
            )
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def set_pose_head(self, cfg, head_type="mlp"):
        self.pose_head = camera_head_factory(
            head_type, "pose", self.backbone, cfg.pose_head
        )
        self.pose_head2 = camera_head_factory(
            head_type, "pose", self.backbone, cfg.pose_head
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_depth_head(self, head_num, decout, img_shape, ray_embedding=None):
        head = getattr(self, f"depth_head{head_num}")
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
        warmup_pts3d: bool = False,
        gaussian_renderer=None,
    ):
        context_image = context.get("image_lr", context["image"])
        target_image = target.get("image_lr", target["image"]) if target is not None else None
        context_image_sr = (
            self.swinir_upsampler(context_image)
            if self.swinir_upsampler is not None
            else context["image"]
        ) # (0，1)

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

        dec_feat, shape, images = out["dec_feat"], out["shape"], out["images"]

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
                gaussians.means,
                "b v (h w) srf spp xyz -> b v h w (srf spp) xyz",
                h=h,
                w=w,
            )  # (b, v, h, w, 1, 3)
            visualization_dump["opacities"] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )  # (b, v, h, w, 1, 1)

        # Flatten LR Gaussians from per-pixel layout to a flat list of anchors.
        lr_gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz  -> b (v r srf spp) xyz"),
            rearrange(
                gaussians.covariances, "b v r srf spp i j  -> b (v r srf spp) i j"
            ),
            rearrange(gaussians.rotations, "b v r srf spp i    -> b (v r srf spp) i"),
            rearrange(gaussians.scales, "b v r srf spp i    -> b (v r srf spp) i"),
            rearrange(
                gaussians.harmonics, "b v r srf spp c sh -> b (v r srf spp) c sh"
            ),
            rearrange(gaussians.opacities, "b v r srf spp      -> b (v r srf spp)"),
        )

        # The LR branch is a frozen scaffold. Everything consumed by the SR
        # branch is detached so the final rendering loss cannot update it.
        lr_gaussians = self._detach_gaussians(lr_gaussians)
        lr_depth = depths_per_view.detach()
        context_extrinsics = context_extrinsics.detach()
        context_intrinsics = context_intrinsics.detach()
        context_sr = context_image_sr.detach()
        context_lr_tokens = dec_feat[-1][:, :v_cxt].detach()

        sr_h, sr_w = context_sr.shape[-2:]
        depth_sr = F.interpolate(
            rearrange(lr_depth, "b v h w -> (b v) 1 h w"),
            size=(sr_h, sr_w),
            mode="bilinear",
            align_corners=False,
        )
        depth_sr = rearrange(
            depth_sr,
            "(b v) 1 h w -> b v h w",
            b=b,
            v=v_cxt,
        )

        # ResUNet exposes aligned LR-grid and SR-grid feature maps. The LR
        # tokens are frozen, while the ResUNet fusion path remains trainable.
        fused_features = self.resunet_token_fusion(
            context_sr,
            context_lr_tokens,
        )

        # Render the frozen LR scaffold at SR resolution. The signed RGB
        # residual is injected into both feature scales before point sampling.
        if gaussian_renderer is None:
            rgb_render_error_256 = torch.zeros_like(context_sr)
        else:
            with torch.no_grad():
                lr_context_render = gaussian_renderer(
                    lr_gaussians,
                    context_extrinsics,
                    context_intrinsics,
                    context["near"].detach(),
                    context["far"].detach(),
                    (sr_h, sr_w),
                    depth_mode=None,
                ).color
                rgb_render_error_256 = lr_context_render - context_sr

        lr_feature_h, lr_feature_w = fused_features["64"].shape[-2:]
        rgb_render_error_64 = F.interpolate(
            rearrange(
                rgb_render_error_256,
                "b v c h w -> (b v) c h w",
            ),
            size=(lr_feature_h, lr_feature_w),
            mode="bilinear",
            align_corners=False,
        )
        rgb_render_error_64 = rearrange(
            rgb_render_error_64,
            "(b v) c h w -> b v c h w",
            b=b,
            v=v_cxt,
        )
        lr_sampling_features = torch.cat(
            (
                fused_features["64"],
                rgb_render_error_64.to(dtype=fused_features["64"].dtype),
            ),
            dim=2,
        )
        sr_sampling_features = torch.cat(
            (
                fused_features["256"],
                rgb_render_error_256.to(dtype=fused_features["256"].dtype),
            ),
            dim=2,
        )

        # Draw the same number of entropy-guided SR pixels in every view and
        # lift them into world space with bilinearly upsampled LR depth.
        sr_points = self.entropy_point_sampler(
            context_sr,
            lr_depth,
            context_extrinsics,
            context_intrinsics,
            deterministic=not self.training,
        )
        initial_sr_gaussians = self._initialize_sr_gaussians(
            sr_points.means,
            sr_points.source_view,
            sr_points.source_pixel_index,
            context_sr,
        )

        lr_points_per_view = h * w
        expected_lr_gaussians = v_cxt * lr_points_per_view
        if lr_gaussians.means.shape[1] != expected_lr_gaussians:
            raise ValueError(
                "The SR branch currently expects one LR Gaussian per input "
                f"pixel, got {lr_gaussians.means.shape[1]} Gaussians for "
                f"{v_cxt}x{h}x{w}={expected_lr_gaussians} pixels."
            )
        lr_source_view = self._source_view_indices(
            b,
            v_cxt,
            lr_points_per_view,
            context_image.device,
        )

        # A point always keeps its source-view feature. Features from another
        # view are added only when projection and depth consistency are valid.
        lr_feature_samples = self.lr_feature_sampler(
            lr_gaussians.means,
            lr_sampling_features,
            context_extrinsics,
            context_intrinsics,
            depth_map=lr_depth,
            source_view=lr_source_view,
            depth_tolerance=self.cfg.multiview_depth_tolerance,
        )
        sr_feature_samples = self.sr_feature_sampler(
            initial_sr_gaussians.means,
            sr_sampling_features,
            context_extrinsics,
            context_intrinsics,
            depth_map=depth_sr,
            source_view=sr_points.source_view,
            depth_tolerance=self.cfg.multiview_depth_tolerance,
        )
        lr_visual_features = self.lr_visual_proj(
            self._aggregate_multiview_features(lr_feature_samples)
        )
        sr_visual_features = self.sr_visual_proj(
            self._aggregate_multiview_features(sr_feature_samples)
        )

        initial_gaussians = self._concatenate_gaussians(
            lr_gaussians,
            initial_sr_gaussians,
        )
        initial_means = initial_gaussians.means.detach()
        point_geometry = self.point_geometry_encoder(initial_means)
        gaussian_parameters = self._pack_gaussian_parameters(initial_gaussians)
        visual_features = torch.cat(
            (lr_visual_features, sr_visual_features),
            dim=1,
        )
        intrinsic_features = self._encode_point_intrinsics(
            context_intrinsics,
            lr_source_view,
            sr_points.source_view,
        )
        point_types = torch.cat(
            (
                torch.zeros_like(lr_gaussians.opacities, dtype=torch.long),
                torch.ones_like(initial_sr_gaussians.opacities, dtype=torch.long),
            ),
            dim=1,
        )
        point_features = torch.cat(
            (
                visual_features,
                gaussian_parameters,
                intrinsic_features,
                self.point_type_embedding(point_types),
            ),
            dim=-1,
        )
        litept_features = self.anchor_litept_fusion(
            initial_means,
            point_features,
            point_geometry,
        )
        refined_means, mean_offsets = self.gaussian_residual_refiner.decode_means(
            litept_features,
            initial_means,
            num_lr_gaussians=lr_gaussians.means.shape[1],
        )
        refined_gaussians = self.gaussian_residual_refiner(
            litept_features,
            refined_means,
            mean_offsets,
            gaussian_parameters,
            num_lr_gaussians=lr_gaussians.means.shape[1],
        )

        num_lr_gaussians = lr_gaussians.means.shape[1]
        refined_lr_gaussians = self._slice_gaussians(
            refined_gaussians,
            0,
            num_lr_gaussians,
        )
        refined_sr_gaussians = self._slice_gaussians(
            refined_gaussians,
            num_lr_gaussians,
            refined_gaussians.means.shape[1],
        )
        encoder_output = {
            "gaussians": refined_gaussians,
            "lr_gaussians": refined_lr_gaussians,
            "sr_gaussians": refined_sr_gaussians,
        }

        if self.cfg.estimating_pose:
            encoder_output["extrinsics"] = dict()
            encoder_output["extrinsics"]["c"] = pred_extrinsics[:, :v_cxt]
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
