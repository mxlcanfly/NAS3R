from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import math

from .backbone.croco.misc import transpose_to_landscape
from .heads.fusion_dpt import create_fusion_dpt
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, inverse_normalize_image, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from ..decoder.cuda_splatting import render_cuda
from .backbone import Backbone, BackboneCfg, get_backbone
from .child_gaussian_feature_decoder import ChildGaussianFeatureDecoder, ChildGaussianFeatureDecoderCfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .gd_style_anchor_densifier import GDStyleAnchorDensifier, GDStyleAnchorDensifierCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector, \
    unproject_depth_map_to_point_map_batch
from .heads.pose_head import PoseHeadCfg
from .lr_anchor_sr_feature_sampler import LRAnchorSRFeatureSampler
from .resunet_fusion import ImageNetResUnetFeatureExtractor
from ..super_resolution import FrozenSwinIRUpsampler

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class SwinIRBranchCfg:
    enabled: bool = False
    weight_path: str = ""
    upscale: int = 4
    img_size: int = 64
    window_size: int = 8
    freeze_lr_branch: bool = True
    share_lr_backbone: bool = False
    output_key: str = "sr_tokens"
    resunet_enabled: bool = True
    resunet_output_key: str = "sr_resunet_features"
    resunet_feature_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    freeze_resunet: bool = False
    fusion_dpt_enabled: bool = True
    fusion_dpt_output_key: str = "sr_fusion_dpt_feature"
    anchor_sampler_enabled: bool = True
    anchor_sampler_patch_size: int = 4
    anchor_sampler_padding_mode: str = "border"
    anchor_sampler_output_key: str = "lr_anchor_sr_features"
    anchor_render_background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    anchor_render_scale_invariant: bool = True
    gd_densifier: GDStyleAnchorDensifierCfg = field(default_factory=GDStyleAnchorDensifierCfg)
    child_gaussian_decoder: ChildGaussianFeatureDecoderCfg = field(default_factory=ChildGaussianFeatureDecoderCfg)


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
    sr_branch: SwinIRBranchCfg = field(default_factory=SwinIRBranchCfg)


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

        self.sr_upsampler = self._build_sr_upsampler(cfg.sr_branch)
        self.sr_backbone = self._build_sr_backbone(cfg.sr_branch)
        self.sr_resunet = self._build_sr_resunet(cfg.sr_branch)
        self.sr_fusion_dpt = self._build_sr_fusion_dpt(cfg.sr_branch)
        self.lr_anchor_sr_sampler = self._build_lr_anchor_sr_sampler(cfg.sr_branch)
        self.gd_anchor_densifier = self._build_gd_anchor_densifier(cfg.sr_branch)
        self.child_gaussian_decoder = self._build_child_gaussian_decoder(cfg.sr_branch)
        self.register_buffer(
            "anchor_render_background_color",
            torch.tensor(cfg.sr_branch.anchor_render_background_color, dtype=torch.float32),
            persistent=False,
        )
        if cfg.sr_branch.enabled and cfg.sr_branch.freeze_lr_branch:
            self.freeze_lr_branch()

    def _build_sr_upsampler(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled:
            return None
        if not cfg.weight_path:
            raise ValueError("sr_branch.weight_path must be set when sr_branch.enabled is true.")
        return FrozenSwinIRUpsampler(
            weight_path=cfg.weight_path,
            upscale=cfg.upscale,
            img_size=cfg.img_size,
            window_size=cfg.window_size,
        )

    def _build_sr_backbone(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled:
            return None
        if cfg.share_lr_backbone:
            return self.backbone
        return deepcopy(self.backbone)

    def _build_sr_resunet(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled or not cfg.resunet_enabled:
            return None

        feature_dims = tuple(cfg.resunet_feature_dims)
        if len(feature_dims) != 3:
            raise ValueError("sr_branch.resunet_feature_dims must contain exactly three channel sizes.")

        resunet = ImageNetResUnetFeatureExtractor(feature_dims=feature_dims)
        if cfg.freeze_resunet:
            self._set_trainable(resunet, False)
        return resunet

    def _build_sr_fusion_dpt(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled or not cfg.fusion_dpt_enabled:
            return None
        if not cfg.resunet_enabled:
            raise ValueError("sr_branch.resunet_enabled must be true when fusion_dpt_enabled is true.")

        feature_dims = tuple(cfg.resunet_feature_dims)
        if len(feature_dims) != 3:
            raise ValueError("sr_branch.resunet_feature_dims must contain exactly three channel sizes.")

        return create_fusion_dpt(self.sr_backbone, resunet_channels=feature_dims)

    def _build_lr_anchor_sr_sampler(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled or not cfg.anchor_sampler_enabled:
            return None
        return LRAnchorSRFeatureSampler(
            patch_size=cfg.anchor_sampler_patch_size,
            padding_mode=cfg.anchor_sampler_padding_mode,
        )

    def _build_gd_anchor_densifier(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled or not cfg.anchor_sampler_enabled or not cfg.gd_densifier.enabled:
            return None
        return GDStyleAnchorDensifier(cfg.gd_densifier)

    def _build_child_gaussian_decoder(self, cfg: SwinIRBranchCfg) -> nn.Module | None:
        if not cfg.enabled or not cfg.child_gaussian_decoder.enabled:
            return None
        if not cfg.fusion_dpt_enabled:
            raise ValueError("sr_branch.fusion_dpt_enabled must be true when child_gaussian_decoder is enabled.")
        return ChildGaussianFeatureDecoder(
            cfg.child_gaussian_decoder,
            d_sh=self.gaussian_adapter.d_sh,
        )

    @staticmethod
    def _get_first_available(data: dict, keys: tuple[str, ...]) -> Tensor | None:
        for key in keys:
            if key in data:
                return data[key]
        return None

    @staticmethod
    def _get_last_dpt_gs_feature(head: nn.Module) -> Tensor:
        dpt = getattr(head, "dpt", None)
        feature = getattr(dpt, "last_path_1", None)
        if feature is None:
            raise RuntimeError(
                "The GS DPT head did not expose last_path_1. "
                "GD-style anchor densification requires dpt_gs_head.py to cache "
                "the feature after `path_1 = path_1 + direct_img_feat`."
            )
        return feature

    def _run_gd_anchor_densifier(
        self,
        anchors: Tensor,
        lr_gs_features: Tensor | None,
        sampled_feature_output: dict,
    ) -> dict:
        if self.gd_anchor_densifier is None:
            return {}
        if lr_gs_features is None:
            return {}

        sampled_features = sampled_feature_output.get(self.cfg.sr_branch.anchor_sampler_output_key)
        if sampled_features is None:
            return {}

        densifier_output = self.gd_anchor_densifier(
            anchors=anchors,
            lr_gs_features=lr_gs_features,
            sampled_features=sampled_features,
        )
        return {f"gd_anchor_{key}": value for key, value in densifier_output.items()}

    def _run_child_gaussian_decoder(
        self,
        sr_branch_output: dict | None,
        gd_anchor_output: dict,
        parent_gaussians: Gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> dict:
        if self.child_gaussian_decoder is None or sr_branch_output is None:
            return {}
        child_centers = gd_anchor_output.get("gd_anchor_child_centers")
        sr_feature_map = sr_branch_output.get(self.cfg.sr_branch.fusion_dpt_output_key)
        if child_centers is None or sr_feature_map is None:
            return {}

        child_output = self.child_gaussian_decoder(
            child_centers=child_centers,
            sr_feature_map=sr_feature_map,
            parent_gaussians=parent_gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        return {
            "child_gaussians": child_output["gaussians"],
            "child_gaussian_features": child_output["features"],
            "child_gaussian_valid": child_output["valid"],
        }

    def _render_lr_gaussians_for_anchor_sampling(
        self,
        gaussians: Gaussians,
        context: dict,
        context_extrinsics: Tensor,
        context_intrinsics: Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, v = context_extrinsics.shape[:2]
        if context["near"].shape[:2] != (b, v) or context["far"].shape[:2] != (b, v):
            raise ValueError("Context near/far must match context camera batch and view axes.")
        color, depth, alpha = render_cuda(
            rearrange(context_extrinsics, "b v i j -> (b v) i j"),
            rearrange(context_intrinsics, "b v i j -> (b v) i j"),
            rearrange(context["near"], "b v -> (b v)"),
            rearrange(context["far"], "b v -> (b v)"),
            image_shape,
            repeat(self.anchor_render_background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
            repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
            scale_invariant=self.cfg.sr_branch.anchor_render_scale_invariant,
            enable_cov_grad=False,
            enable_sh_grad=False,
            return_alpha=True,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) 1 h w -> b v h w", b=b, v=v)
        if self.cfg.sr_branch.anchor_render_scale_invariant:
            depth = depth * context["near"][:, :, None, None]
        return color, depth, alpha

    def _sample_lr_anchor_sr_features(
        self,
        sr_branch_output: dict | None,
        context: dict,
        gaussians: Gaussians,
        anchors: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> dict:
        if self.lr_anchor_sr_sampler is None or sr_branch_output is None:
            return {}

        sr_image = sr_branch_output.get("sr_image")
        render_error = self._get_first_available(
            context,
            ("render_error_sr", "rendered_error_sr"),
        )
        if sr_image is None:
            return {}

        with torch.no_grad():
            render_color, render_depth, render_alpha = self._render_lr_gaussians_for_anchor_sampling(
                gaussians=gaussians,
                context=context,
                context_extrinsics=extrinsics,
                context_intrinsics=intrinsics,
                image_shape=sr_image.shape[-2:],
            )
            if render_error is None:
                render_error = sr_image - render_color
            feature_stack = self.lr_anchor_sr_sampler.build_feature_stack(
                sr_image=sr_image,
                render_color=render_color,
                render_depth=render_depth,
                render_alpha=render_alpha,
                render_error=render_error,
            )
            sampled_features = self.lr_anchor_sr_sampler(
                anchors=anchors,
                feature_stack=feature_stack,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
            )
        return {
            self.cfg.sr_branch.anchor_sampler_output_key: sampled_features,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        has_sr_backbone = any(key.startswith("sr_backbone.") for key in state_dict)
        if self.sr_backbone is not None and self.sr_backbone is not self.backbone and not has_sr_backbone:
            state_dict = dict(state_dict)
            for key, value in list(state_dict.items()):
                if key.startswith("backbone."):
                    state_dict[f"sr_{key}"] = value
        result = super().load_state_dict(state_dict, strict=strict)
        if self.sr_backbone is not None and self.sr_backbone is not self.backbone and not has_sr_backbone:
            self.sr_backbone.load_state_dict(self.backbone.state_dict())
        return result

    @staticmethod
    def _set_trainable(module: nn.Module | None, trainable: bool) -> None:
        if module is None:
            return
        if not isinstance(module, nn.Module):
            return
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def freeze_lr_branch(self) -> None:
        """Freeze the original LR path while leaving added SR modules configurable."""
        lr_modules = [
            self.backbone,
            getattr(self, "downstream_depth_head1", None),
            getattr(self, "downstream_depth_head2", None),
            getattr(self, "gaussian_param_head", None),
            getattr(self, "gaussian_param_head2", None),
            getattr(self, "pose_head", None),
            getattr(self, "pose_head2", None),
        ]
        for module in lr_modules:
            self._set_trainable(module, False)

    def _sr_target_size(self, context: dict, target: Optional[dict]) -> tuple[int, int]:
        if target is not None and "image" in target:
            return target["image"].shape[-2:]
        return context["image"].shape[-2:]

    def _make_sr_context_image(
        self,
        context_image: Tensor,
        target_size: tuple[int, int],
    ) -> Tensor:
        if self.sr_upsampler is None:
            raise RuntimeError("SR upsampler is not initialized.")

        sr_image = self.sr_upsampler(context_image)
        if sr_image.shape[-2:] != target_size:
            *batch, c, h, w = sr_image.shape
            sr_image = rearrange(sr_image, "... c h w -> (...) c h w")
            sr_image = F.interpolate(sr_image, size=target_size, mode="bicubic", align_corners=False)
            sr_image = sr_image.reshape(*batch, c, *target_size)
        return sr_image.clamp(0, 1)

    def _swinir_input_image(self, context: dict, context_image: Tensor) -> Tensor:
        if "image_lr" in context:
            return context_image
        return inverse_normalize_image(context_image, self.cfg.input_mean, self.cfg.input_std).clamp(0, 1)

    def _encode_sr_context(
        self,
        context: dict,
        context_image: Tensor,
        target: Optional[dict],
    ) -> dict:
        target_size = self._sr_target_size(context, target)
        sr_image = self._make_sr_context_image(self._swinir_input_image(context, context_image), target_size)
        sr_context = {
            "image": normalize_image(sr_image),
            "intrinsics": context["intrinsics"],
        }
        sr_out = self.sr_backbone(sr_context, target_num_views=0)
        output = {
            self.cfg.sr_branch.output_key: sr_out["dec_feat"],
            "sr_image": sr_image,
            "sr_shape": sr_out["shape"],
        }
        sr_resunet_features = None
        if self.sr_resunet is not None:
            sr_resunet_features = self.sr_resunet(sr_image)
            output[self.cfg.sr_branch.resunet_output_key] = sr_resunet_features
        if self.sr_fusion_dpt is not None:
            output[self.cfg.sr_branch.fusion_dpt_output_key] = self.sr_fusion_dpt(
                sr_out["dec_feat"],
                sr_resunet_features,
                target_size,
            )
        return output

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

        sr_branch_output = None
        if self.sr_upsampler is not None:
            sr_branch_output = self._encode_sr_context(context, context_image, target)

        dec_feat, shape, images = out['dec_feat'], out['shape'], out['images']

        with torch.amp.autocast('cuda', enabled=False):
            all_other_params = []
            all_depth_res = []
            all_lr_gs_features = []

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
                all_lr_gs_features.append(
                    rearrange(self._get_last_dpt_gs_feature(self.gaussian_param_head), "b c h w -> b (h w) c")
                )
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v_cxt):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3],
                                                        shape[0, i].cpu().tolist())
                    all_lr_gs_features.append(
                        rearrange(self._get_last_dpt_gs_feature(self.gaussian_param_head2), "b c h w -> b (h w) c")
                    )
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
        lr_gs_features = torch.stack(all_lr_gs_features, dim=1) if all_lr_gs_features else None
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
        lr_anchors = depth_to_pts_all
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
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )  # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )  # (b, v, h, w, 1, 1)

        encoder_output = dict()

        flat_gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )
        encoder_output["gaussians"] = flat_gaussians
        lr_anchor_sr_features = self._sample_lr_anchor_sr_features(
            sr_branch_output,
            context,
            flat_gaussians,
            lr_anchors,
            context_extrinsics,
            context_intrinsics,
        )
        gd_anchor_output = self._run_gd_anchor_densifier(
            anchors=lr_anchors,
            lr_gs_features=lr_gs_features,
            sampled_feature_output=lr_anchor_sr_features,
        )
        child_gaussian_output = self._run_child_gaussian_decoder(
            sr_branch_output=sr_branch_output,
            gd_anchor_output=gd_anchor_output,
            parent_gaussians=flat_gaussians,
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
        )
        if child_gaussian_output:
            encoder_output["lr_gaussians"] = flat_gaussians
            encoder_output["gaussians"] = child_gaussian_output["child_gaussians"]

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

        if sr_branch_output is not None:
            encoder_output.update(sr_branch_output)
        encoder_output.update(lr_anchor_sr_features)
        encoder_output.update(gd_anchor_output)
        encoder_output.update(child_gaussian_output)

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
