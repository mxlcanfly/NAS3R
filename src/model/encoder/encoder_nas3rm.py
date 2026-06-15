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
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector, \
    unproject_depth_map_to_point_map_batch
from .heads.pose_head import PoseHeadCfg
from .resunet_fusion import HiSplatResUnetTokenFusion
from ..super_resolution import FrozenSwinIRUpsampler
from ..decoder.cuda_splatting import render_cuda

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
    compute_render_error: bool = False
    render_error_background: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    use_resunet_token_fusion: bool = False
    unimatch_weights_path: str = ""
    densification_grid_size: int = 64
    densification_budget: int = 2048
    densification_sampling: Literal["topk", "multinomial"] = "multinomial"


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
        if cfg.compute_render_error and self.swinir is None:
            raise ValueError(
                "compute_render_error=True requires use_swinir=True"
            )
        if cfg.use_resunet_token_fusion and self.swinir is None:
            raise ValueError(
                "use_resunet_token_fusion=True requires use_swinir=True"
            )
        self.resunet_token_fusion = (
            HiSplatResUnetTokenFusion(
                token_dim=self.backbone.dec_embed_dim,
            )
            if cfg.use_resunet_token_fusion
            else None
        )
        if (
            self.resunet_token_fusion is not None
            and cfg.unimatch_weights_path
        ):
            self.resunet_token_fusion.load_unimatch_encoder(
                cfg.unimatch_weights_path
            )
        self.register_buffer(
            "render_error_background",
            torch.tensor(cfg.render_error_background, dtype=torch.float32),
            persistent=False,
        )

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
        expected_size = (self.cfg.swinir_input_size, self.cfg.swinir_input_size)
        if images.shape[-2:] != expected_size:
            raise ValueError(
                f"SwinIR expects {expected_size} inputs, got {tuple(images.shape[-2:])}"
            )
        return self.swinir(images)

    @torch.no_grad()
    def _compute_render_error(
            self,
            gaussians: Gaussians,
            sr_images: Tensor,
            extrinsics: Tensor,
            intrinsics: Tensor,
            near: Tensor,
            far: Tensor,
    ) -> dict[str, Tensor]:
        b, v, _, h, w = sr_images.shape
        rendered_color, _, accumulated_opacity = render_cuda(
            rearrange(extrinsics.detach(), "b v i j -> (b v) i j"),
            rearrange(intrinsics.detach(), "b v i j -> (b v) i j"),
            rearrange(near.detach(), "b v -> (b v)"),
            rearrange(far.detach(), "b v -> (b v)"),
            (h, w),
            repeat(
                self.render_error_background,
                "c -> (b v) c",
                b=b,
                v=v,
            ),
            repeat(gaussians.means.detach(), "b g xyz -> (b v) g xyz", v=v),
            repeat(
                gaussians.covariances.detach(),
                "b g i j -> (b v) g i j",
                v=v,
            ),
            repeat(
                gaussians.harmonics.detach(),
                "b g c d -> (b v) g c d",
                v=v,
            ),
            repeat(gaussians.opacities.detach(), "b g -> (b v) g", v=v),
            repeat(gaussians.rotations.detach(), "b g d -> (b v) g d", v=v),
            repeat(gaussians.scales.detach(), "b g d -> (b v) g d", v=v),
            scale_invariant=True,
            enable_cov_grad=False,
            enable_sh_grad=False,
        )
        rendered_color = rearrange(
            rendered_color,
            "(b v) c h w -> b v c h w",
            b=b,
            v=v,
        )
        accumulated_opacity = rearrange(
            accumulated_opacity,
            "(b v) 1 h w -> b v 1 h w",
            b=b,
            v=v,
        )
        sr_images = sr_images.detach()
        return {
            "lr_render": rendered_color,
            "render_error": (rendered_color - sr_images),
            "accumulated_opacity": accumulated_opacity,
        }

    @torch.no_grad()
    def _compute_densification_mask(
            self,
            render_error: Tensor,
    ) -> dict[str, Tensor]:
        if render_error.ndim != 5:
            raise ValueError(
                "render_error must have shape [B, V, C, H, W], got "
                f"{tuple(render_error.shape)}"
            )

        b, v = render_error.shape[:2]
        grid_size = self.cfg.densification_grid_size
        if grid_size <= 0:
            raise ValueError(
                f"densification_grid_size must be positive, got {grid_size}"
            )
        error_score = render_error.detach().abs().mean(dim=2)
        error_score = F.adaptive_avg_pool2d(
            rearrange(error_score, "b v h w -> (b v) 1 h w"),
            (grid_size, grid_size),
        )
        error_score = rearrange(
            error_score,
            "(b v) 1 h w -> b v h w",
            b=b,
            v=v,
        )

        flat_score = rearrange(
            error_score.float(),
            "b v h w -> b v (h w)",
        )
        sampling_weight = flat_score + 1e-8
        probability = sampling_weight / sampling_weight.sum(
            dim=-1,
            keepdim=True,
        )

        num_candidates = probability.shape[-1]
        budget = min(max(int(self.cfg.densification_budget), 0), num_candidates)
        flat_mask = torch.zeros_like(probability, dtype=torch.bool)
        if budget > 0:
            if self.cfg.densification_sampling == "multinomial":
                selected = torch.multinomial(
                    probability.reshape(b * v, num_candidates),
                    num_samples=budget,
                    replacement=False,
                )
                selected = rearrange(selected, "(b v) k -> b v k", b=b, v=v)
            elif self.cfg.densification_sampling == "topk":
                selected = probability.topk(budget, dim=-1).indices
            else:
                raise ValueError(
                    "densification_sampling must be 'topk' or 'multinomial', "
                    f"got {self.cfg.densification_sampling!r}"
                )
            flat_mask.scatter_(-1, selected, True)

        densification_mask = rearrange(
            flat_mask,
            "b v (h w) -> b v h w",
            h=grid_size,
            w=grid_size,
        )
        return {
            "densification_probability": rearrange(
                probability,
                "b v (h w) -> b v h w",
                h=grid_size,
                w=grid_size,
            ),
            "densification_mask": densification_mask,
            "remaining_mask": ~densification_mask,
        }

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
        warmup_pts3d: bool = False,
    ):
        context_image_lr = context.get("image_lr", context["image"])
        target_image_lr = (
            target.get("image_lr", target["image"])
            if target is not None
            else None
        )
        context_image_sr = (
            self._super_resolve(context_image_lr)
            if self.cfg.compute_render_error
            else None
        )

        b, v_cxt, _, h, w = context_image_lr.shape

        if target is not None:
            v_tgt = target_image_lr.shape[1]
            context_target = {
                "image": normalize_image(
                    torch.cat([context_image_lr, target_image_lr], dim=1)
                ),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1),
            }
            # Encode the context and target images.
            out = self.backbone(context_target, target_num_views=v_tgt)
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context_image_lr),
                "intrinsics": context["intrinsics"],
            }
            # Encode the context images.
            out = self.backbone(context_input)

        dec_feat, shape, images = out['dec_feat'], out['shape'], out['images']
        resunet_features = (
            self.resunet_token_fusion(
                context_image_sr,
                dec_feat[-1][:, :v_cxt],
            )
            if self.resunet_token_fusion is not None
            else None
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
                GS_res1, gs_feat1 = self.gaussian_param_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3],
                                                   shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v_cxt):
                    GS_res2, gs_feat2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3],
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
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )  # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )  # (b, v, h, w, 1, 1)

        encoder_output = dict()

        output_gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )
        encoder_output["gaussians"] = output_gaussians
        if resunet_features is not None:
            encoder_output["resunet_features"] = resunet_features

        if self.cfg.compute_render_error:
            if "near" not in context or "far" not in context:
                raise KeyError(
                    "context must contain near and far to compute render_error"
                )
            render_error_output = self._compute_render_error(
                output_gaussians,
                context_image_sr,
                context_extrinsics,
                context_intrinsics,
                context["near"],
                context["far"],
            )
            encoder_output.update(
                self._compute_densification_mask(
                    render_error_output["render_error"]
                )
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
