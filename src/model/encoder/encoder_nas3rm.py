from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
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
from ..utils import (
    build_lr_context_feature_stack,
    GDCrossAttentionCfg,
    GDGaussianFeatureCrossAttention,
    GDStyleGaussianChildDecoder,
    GDStyleGaussianChildDecoderCfg,
    HiSplatSingleViewFeatureCfg,
    ReSplatGaussianPointTransformer,
    ReSplatPointTransformerCfg,
    render_gaussians_to_context,
    sample_lr_gaussian_point_features,
    SingleViewSRFeatureExtractor,
)
from ..super_resolution import FrozenSwinIRUpsampler

inf = float('inf')


def _write_pixel_grid_2d_debug_image(
    image_shape: tuple[int, int],
    path: Path,
    max_cells: int = 16,
    initial_child_offsets_uv: Tensor | None = None,
    updated_child_offsets_uv: Tensor | None = None,
    parents_per_pixel: int = 1,
) -> None:
    """Compare initial and predicted child UV positions on the LR pixel grid."""
    import matplotlib.pyplot as plt

    h, w = image_shape
    cells_h = min(h, max_cells)
    cells_w = min(w, max_cells)
    xs = torch.arange(cells_w, dtype=torch.float32)
    ys = torch.arange(cells_h, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    current_x = grid_x / w
    current_y = grid_y / h

    if initial_child_offsets_uv is None:
        initial_child_offsets_uv = torch.empty(0, 2)
    initial_child_offsets_uv = initial_child_offsets_uv.detach().float().cpu()
    radii = initial_child_offsets_uv.square().sum(dim=-1).sqrt()
    ring_radii = torch.unique(radii.round(decimals=5), sorted=True)
    ring_masks = [
        torch.isclose(radii, ring_radius, atol=1e-4, rtol=0.0)
        for ring_radius in ring_radii
    ]

    if updated_child_offsets_uv is not None:
        expected_points = h * w * parents_per_pixel
        if updated_child_offsets_uv.shape[:1] != (expected_points,):
            raise ValueError(
                "updated_child_offsets_uv must contain one view of LR points: "
                f"expected first dimension {expected_points}, got {tuple(updated_child_offsets_uv.shape)}"
            )
        updated_child_offsets_uv = rearrange(
            updated_child_offsets_uv.detach().float().cpu(),
            "(h w q) k xy -> h w q k xy",
            h=h,
            w=w,
            q=parents_per_pixel,
        )[:cells_h, :cells_w]
        offset_delta_l2 = (
            updated_child_offsets_uv - initial_child_offsets_uv[None, None, None]
        ).norm(dim=-1)
        updated_title = (
            "Updated XY after bias + predicted residual\n"
            f"delta L2 in LR pixels: mean={offset_delta_l2.mean():.4f}, "
            f"max={offset_delta_l2.amax():.4f}"
        )

    def draw_panel(ax, offsets_uv: Tensor, title: str) -> None:
        for x in range(cells_w + 1):
            ax.axvline(x / w, color="0.82", linewidth=1)
        for y in range(cells_h + 1):
            ax.axhline(y / h, color="0.82", linewidth=1)
        ax.scatter(
            current_x.flatten(),
            current_y.flatten(),
            c="#e3342f",
            s=30,
            label="parent: u/W, v/H",
            zorder=4,
        )
        colors = plt.get_cmap("tab10")
        for ring_idx, ring_mask in enumerate(ring_masks):
            if offsets_uv.ndim == 2:
                ring_offsets = offsets_uv[ring_mask]
                child_x = current_x[..., None] + ring_offsets[:, 0] / w
                child_y = current_y[..., None] + ring_offsets[:, 1] / h
            else:
                ring_offsets = offsets_uv[..., ring_mask, :]
                child_x = current_x[..., None, None] + ring_offsets[..., 0] / w
                child_y = current_y[..., None, None] + ring_offsets[..., 1] / h
            ax.scatter(
                child_x.flatten(),
                child_y.flatten(),
                color=colors(ring_idx % 10),
                s=10,
                label=f"child ring {ring_idx + 1}",
                zorder=3,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.25 / w, (cells_w + 0.75) / w)
        ax.set_ylim((cells_h + 0.75) / h, -0.25 / h)
        ax.set_xlabel("normalized x")
        ax.set_ylabel("normalized y")
        ax.set_title(title)
        ax.legend(loc="upper right")

    def draw_single_pixel_panel(ax, offsets_uv: Tensor) -> None:
        pixel_y = cells_h // 2
        pixel_x = cells_w // 2
        updated = offsets_uv[pixel_y, pixel_x, 0]
        if updated.shape != initial_child_offsets_uv.shape:
            raise ValueError(
                "Initial/updated child count mismatch in UV visualization: "
                f"{tuple(initial_child_offsets_uv.shape)} vs {tuple(updated.shape)}"
            )
        colors = plt.get_cmap("tab10")
        for child_idx, (initial_uv, updated_uv) in enumerate(
            zip(initial_child_offsets_uv, updated)
        ):
            color = colors(child_idx % 10)
            ax.annotate(
                "",
                xy=updated_uv.tolist(),
                xytext=initial_uv.tolist(),
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.2},
                zorder=2,
            )
            ax.scatter(
                initial_uv[0],
                initial_uv[1],
                marker="x",
                color=color,
                s=45,
                zorder=3,
            )
            ax.scatter(
                updated_uv[0],
                updated_uv[1],
                marker="o",
                color=color,
                edgecolors="white",
                linewidths=0.7,
                s=55,
                zorder=4,
            )
            ax.annotate(
                str(child_idx + 1),
                updated_uv.tolist(),
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=9,
                weight="bold",
                zorder=5,
            )
        ax.scatter(0.0, 0.0, marker="+", color="#e3342f", s=90, linewidths=2, zorder=6)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(1.03, -0.03)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="0.85", linewidth=1)
        ax.set_xlabel("local offset u in LR pixels")
        ax.set_ylabel("local offset v in LR pixels")
        ax.set_title(
            f"Single-pixel child motion at LR pixel ({pixel_x}, {pixel_y})\n"
            "x = initialization, o = updated, arrow = motion"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    panel_count = 3 if updated_child_offsets_uv is not None else 1
    fig, axes = plt.subplots(1, panel_count, figsize=(7 * panel_count, 7), squeeze=False)
    draw_panel(axes[0, 0], initial_child_offsets_uv, "Initial XY from quarter-ring bias")
    if updated_child_offsets_uv is not None:
        draw_panel(
            axes[0, 1],
            updated_child_offsets_uv,
            updated_title,
        )
        draw_single_pixel_panel(axes[0, 2], updated_child_offsets_uv)
    fig.suptitle(f"LR pixel grid: showing {cells_h}x{cells_w} of {h}x{w}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    gd_cross_attn: GDCrossAttentionCfg = field(default_factory=GDCrossAttentionCfg)
    resplat_pt: ReSplatPointTransformerCfg = field(default_factory=ReSplatPointTransformerCfg)
    gaussian_child_decoder: GDStyleGaussianChildDecoderCfg = field(default_factory=GDStyleGaussianChildDecoderCfg)
    freeze_original_lr_network: bool = True
    trainable_new_modules: list[str] = field(default_factory=lambda: [
        "gd_cross_attn",
        "resplat_pt",
        "gaussian_child_decoder",
        "hisplat_sr_feature_extractor",
    ])
    swinir_weight_path: str = "/space0/mengxl/NAS3R-master/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    swinir_upscale: int = 4
    swinir_img_size: int = 64
    swinir_window_size: int = 8
    hisplat_sr_features: HiSplatSingleViewFeatureCfg = field(default_factory=HiSplatSingleViewFeatureCfg)


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
        self.gd_cross_attn = (
            GDGaussianFeatureCrossAttention(cfg.gd_cross_attn)
            if cfg.gd_cross_attn.enabled
            else None
        )
        self.resplat_pt_input_dim = cfg.gd_cross_attn.hidden_dim * 2
        self.resplat_pt = (
            ReSplatGaussianPointTransformer(cfg.resplat_pt, input_dim=self.resplat_pt_input_dim)
            if cfg.resplat_pt.enabled
            else None
        )
        self.gaussian_child_decoder = (
            GDStyleGaussianChildDecoder(
                cfg.gaussian_child_decoder,
                sh_degree=cfg.gaussian_adapter.sh_degree,
            )
            if cfg.gaussian_child_decoder.enabled
            else None
        )
        self.swinir_upsampler = FrozenSwinIRUpsampler(
            cfg.swinir_weight_path,
            upscale=cfg.swinir_upscale,
            img_size=cfg.swinir_img_size,
            window_size=cfg.swinir_window_size,
        )
        self.hisplat_sr_feature_extractor = (
            SingleViewSRFeatureExtractor(cfg.hisplat_sr_features)
            if cfg.hisplat_sr_features.enabled
            else None
        )

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

        if cfg.freeze_original_lr_network:
            self.freeze_original_lr_network()

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

    def run_resplat_pt(
            self,
            points: Tensor,
            features: Tensor,
            offsets: Optional[Tensor] = None,
    ) -> Tensor:
        if self.resplat_pt is None:
            raise RuntimeError("ReSplat point transformer is disabled in cfg.resplat_pt.")
        return self.resplat_pt(points, features, offsets=offsets)

    def freeze_original_lr_network(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

        for module_name in self.cfg.trainable_new_modules:
            module = getattr(self, module_name, None)
            if module is not None:
                module.requires_grad_(True)

        # SwinIR is a fixed SR prior in this stage even though it is newly attached.
        self.swinir_upsampler.requires_grad_(False)
        self.swinir_upsampler.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_original_lr_network:
            trainable = set(self.cfg.trainable_new_modules)
            for module_name, module in self.named_children():
                if module_name not in trainable:
                    module.eval()
            for module_name in trainable:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.train(mode)
            self.swinir_upsampler.eval()
        return self

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
        context_image_sr = self.swinir_upsampler(context_image)
        context_sr_features = None
        if self.hisplat_sr_feature_extractor is not None:
            context_sr_features = self.hisplat_sr_feature_extractor(
                context_image,
                context_image_sr,
                self.backbone,
                croco_image_sr=normalize_image(context_image_sr),
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

        with torch.amp.autocast('cuda', enabled=False):
            all_other_params = []
            all_depth_res = []
            all_gaussian_img_feats = []

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
                all_gaussian_img_feats.append(self.gaussian_param_head.dpt.last_point_feat)
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1)
                for i in range(1, v_cxt):
                    GS_res2 = self.gaussian_param_head2([tok[:, i].float() for tok in dec_feat], images[:, i, :3],
                                                        shape[0, i].cpu().tolist())
                    all_gaussian_img_feats.append(self.gaussian_param_head2.dpt.last_point_feat)
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
        gaussian_img_feats = torch.stack(all_gaussian_img_feats, dim=1)
        lr_gaussian_img_feats = rearrange(
            gaussian_img_feats,
            "b v c h w -> b v (h w) c",
            h=h,
            w=w,
        )
        _, _, _, num_surfaces_actual, gaussians_per_pixel_actual, _ = gaussians.means.shape
        lr_gaussian_img_feats = repeat(
            lr_gaussian_img_feats,
            "b v r c -> b v (r srf spp) c",
            srf=num_surfaces_actual,
            spp=gaussians_per_pixel_actual,
        )
        flat_lr_gaussians = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.rotations, "b v r srf spp i -> b (v r srf spp) i"),
            rearrange(gaussians.scales, "b v r srf spp i -> b (v r srf spp) i"),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )
        lr_context_render = render_gaussians_to_context(
            flat_lr_gaussians,
            context_extrinsics,
            context_intrinsics,
            context["near"],
            context["far"],
            (h, w),
        )
        lr_context_feature_stack = build_lr_context_feature_stack(
            context_image,
            lr_context_render,
        )
        lr_gaussian_points = rearrange(
            gaussians.means,
            "b v r srf spp xyz -> b v (r srf spp) xyz",
        )
        num_lr_gaussian_points = lr_gaussian_points.shape[2]
        children_per_pixel = num_surfaces_actual * gaussians_per_pixel_actual
        pixel_u, pixel_v = torch.meshgrid(
            torch.arange(w, device=device, dtype=depth_all.dtype),
            torch.arange(h, device=device, dtype=depth_all.dtype),
            indexing="xy",
        )
        parent_uv_per_pixel = torch.stack((pixel_u / w, pixel_v / h), dim=-1)
        parent_uv_per_view = repeat(
            parent_uv_per_pixel,
            "h w xy -> (h w q) xy",
            q=children_per_pixel,
        )
        parent_uv = repeat(parent_uv_per_view, "n xy -> b (v n) xy", b=b, v=v_cxt)
        parent_depths = repeat(
            depth_all,
            "b v h w -> b (v h w q)",
            q=children_per_pixel,
        )
        lr_gaussian_point_feats = sample_lr_gaussian_point_features(
            lr_gaussian_points,
            lr_context_feature_stack,
            context_extrinsics,
            context_intrinsics,
        )
        lr_gaussian_cross_attn = None
        lr_gaussian_pt_feats = None
        lr_child_decode = None
        if self.gd_cross_attn is not None:
            lr_gaussian_cross_attn = self.gd_cross_attn(
                lr_gaussian_img_feats,
                lr_gaussian_point_feats,
            )
            lr_gaussian_pt_input_feats = lr_gaussian_cross_attn["pt_input_feats"]
            if self.resplat_pt is not None:
                lr_gaussian_pt_feats = self.resplat_pt(
                    rearrange(lr_gaussian_points, "b v n xyz -> b (v n) xyz"),
                    rearrange(lr_gaussian_pt_input_feats, "b v n c -> b (v n) c"),
                )
                lr_gaussian_pt_feats = rearrange(
                    lr_gaussian_pt_feats,
                    "(b n) c -> b n c",
                    b=b,
                )
                if self.gaussian_child_decoder is not None:
                    lr_child_decode = self.gaussian_child_decoder(
                        flat_lr_gaussians.means,
                        lr_gaussian_pt_feats,
                        flat_lr_gaussians,
                        parent_uv=parent_uv,
                        parent_depths=parent_depths,
                        extrinsics=context_extrinsics,
                        intrinsics=context_intrinsics,
                        image_shape=(h, w),
                        points_per_view=num_lr_gaussian_points,
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
        encoder_output["image_sr"] = context_image_sr
        encoder_output["lr_gaussian_img_feats"] = lr_gaussian_img_feats
        encoder_output["lr_context_render"] = {
            "color": lr_context_render.color.detach(),
            "alpha": lr_context_render.alpha.detach(),
            "depth": lr_context_render.depth.detach(),
        }
        encoder_output["lr_dpt_depth"] = depths_per_view.detach()
        encoder_output["lr_gaussian_point_feats"] = lr_gaussian_point_feats
        if lr_gaussian_cross_attn is not None:
            encoder_output["lr_gaussian_cross_attn"] = lr_gaussian_cross_attn
        if lr_gaussian_pt_feats is not None:
            encoder_output["lr_gaussian_pt_feats"] = lr_gaussian_pt_feats
        if lr_child_decode is not None:
            encoder_output["lr_child_gaussians"] = lr_child_decode["gaussians"]
            encoder_output["lr_child_gaussian_features"] = lr_child_decode["features"]
            encoder_output["lr_child_decode"] = {
                "delta_means": lr_child_decode["delta_means"],
                "local_offset_uv": lr_child_decode["local_offset_uv"],
                "predicted_delta_z": lr_child_decode["predicted_delta_z"],
                "child_depths": lr_child_decode["child_depths"],
                "delta_attrs": lr_child_decode["delta_attrs"],
            }
        if context_sr_features is not None:
            encoder_output["sr_features"] = context_sr_features

        encoder_output["gaussians"] = flat_lr_gaussians

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

    def predict_dpt_depth(self, images: Tensor, intrinsics: Tensor) -> Tensor:
        """Predict DPT z-depth without running Gaussian or child branches."""
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B,V,3,H,W], got {tuple(images.shape)}")
        b, num_views, _, _, _ = images.shape
        backbone_output = self.backbone(
            {
                "image": normalize_image(images),
                "intrinsics": intrinsics,
            }
        )
        dec_feat = backbone_output["dec_feat"]
        shape = backbone_output["shape"]

        depth_outputs = []
        with torch.amp.autocast("cuda", enabled=False):
            depth_outputs.append(
                self._downstream_depth_head(
                    1,
                    [tokens[:, 0].float() for tokens in dec_feat],
                    shape[:, 0],
                )["depth"]
            )
            for view_idx in range(1, num_views):
                depth_outputs.append(
                    self._downstream_depth_head(
                        2,
                        [tokens[:, view_idx].float() for tokens in dec_feat],
                        shape[:, view_idx],
                    )["depth"]
                )
        return torch.stack(depth_outputs, dim=1).squeeze(-1)

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
