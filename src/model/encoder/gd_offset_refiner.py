from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ..types import Gaussians
from .common.gaussians import build_covariance
from ...geometry.projection import (
    homogenize_points,
    project_camera_space,
    transform_world2cam,
)


@dataclass
class GDOffsetRefinerCfg:
    enabled: bool = False
    gs_feature_dim: int = 256
    hidden_dim: int = 160
    sample_patch_size: int = 4
    cross_attn_heads: int = 16
    point_transformer_depth: int = 2
    point_transformer_heads: int = 8
    point_transformer_knn: int = 16
    num_offsets: int = 8
    sr_feature_dim: int = 128
    child_sample_patch_size: int = 2
    child_self_attn_layers: int = 4
    sh_degree: int = 4
    child_scale_divisor: float = 2.0
    child_aggregation_views: int = 2
    child_fused_dim: int = 256
    scale_residual_scale: float = 0.25


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.init(self.net)

    @staticmethod
    def init(layers: nn.Module) -> None:
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class KNNPointTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, knn: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.knn = knn
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.rel_pos = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, coords: Tensor, feat: Tensor) -> Tensor:
        b, n, c = feat.shape
        k = min(self.knn, n)
        x = self.norm1(feat)
        dist = torch.cdist(coords.float(), coords.float())
        knn_idx = dist.topk(k=k, dim=-1, largest=False).indices
        batch_idx = torch.arange(b, device=feat.device)[:, None, None]

        neigh_x = x[batch_idx, knn_idx]
        neigh_coord = coords[batch_idx, knn_idx]
        rel = coords[:, :, None] - neigh_coord
        rel_embed = self.rel_pos(rel)

        query = self.q(x).view(b, n, self.num_heads, self.head_dim)
        key = (self.k(neigh_x) + rel_embed).view(
            b, n, k, self.num_heads, self.head_dim
        )
        value = (self.v(neigh_x) + rel_embed).view(
            b, n, k, self.num_heads, self.head_dim
        )
        attn = (
            query[:, :, None] * key
        ).sum(dim=-1).permute(0, 1, 3, 2) * self.scale
        attn = attn.softmax(dim=-1)
        update = (attn.permute(0, 1, 3, 2)[..., None] * value).sum(dim=2)
        update = update.reshape(b, n, c)
        feat = feat + self.proj(update)
        feat = feat + self.mlp(self.norm2(feat))
        return feat


class KNNPointTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, knn: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                KNNPointTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    knn=knn,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, coords: Tensor, feat: Tensor) -> Tensor:
        for block in self.blocks:
            feat = block(coords, feat)
        return feat


class GDOffsetRefiner(nn.Module):
    def __init__(self, cfg: GDOffsetRefinerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gs_mlp = MLP(cfg.gs_feature_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.gs_norm = nn.LayerNorm(cfg.hidden_dim)
        self.sample_mlp = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.cross_attn_heads,
            kdim=cfg.hidden_dim,
            vdim=cfg.hidden_dim,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.post_attn = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.point_transformer = KNNPointTransformer(
            dim=cfg.hidden_dim,
            depth=cfg.point_transformer_depth,
            num_heads=cfg.point_transformer_heads,
            knn=cfg.point_transformer_knn,
        )
        self.child_feature_fusion = nn.Sequential(
            nn.LayerNorm(
                cfg.child_aggregation_views
                * cfg.child_sample_patch_size ** 2
                * (cfg.sr_feature_dim + 3)
            ),
            nn.Linear(
                cfg.child_aggregation_views
                * cfg.child_sample_patch_size ** 2
                * (cfg.sr_feature_dim + 3),
                cfg.child_fused_dim,
            ),
            nn.GELU(),
            nn.Linear(cfg.child_fused_dim, cfg.child_fused_dim),
        )
        self.child_self_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.child_fused_dim,
                    nhead=cfg.point_transformer_heads,
                    dim_feedforward=cfg.child_fused_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.child_self_attn_layers)
            ]
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_offsets * 3),
        )
        self.attribute_dim = 3 + 1 + 4 + 3 * ((cfg.sh_degree + 1) ** 2)
        self.attribute_head = nn.Sequential(
            nn.LayerNorm(cfg.child_fused_dim),
            nn.Linear(cfg.child_fused_dim, cfg.child_fused_dim),
            nn.GELU(),
            nn.Linear(cfg.child_fused_dim, cfg.child_fused_dim),
            nn.GELU(),
            nn.Linear(cfg.child_fused_dim, self.attribute_dim),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.attribute_head[-1].weight)
        nn.init.zeros_(self.attribute_head[-1].bias)
        for param in self.attribute_head.parameters():
            param._no_weight_decay = True

    def _patch_offsets(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        radius = (patch_size - 1) / 2
        offsets = torch.arange(
            patch_size,
            device=device,
            dtype=dtype,
        ) - radius
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _project(self, anchors: Tensor, extrinsics: Tensor, intrinsics: Tensor):
        b, v = extrinsics.shape[:2]
        anchors_per_view = rearrange(anchors, "b v n xyz -> b v n xyz")
        cam_points = transform_world2cam(
            homogenize_points(anchors_per_view),
            extrinsics[:, :, None],
        )[..., :3]
        uv = project_camera_space(cam_points, intrinsics[:, :, None])
        valid = (
            (cam_points[..., 2] > 1e-6)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= 1)
        )
        return uv, cam_points[..., 2], valid

    def _project_points_to_views(self, points: Tensor, extrinsics: Tensor, intrinsics: Tensor):
        cam_points = transform_world2cam(
            homogenize_points(points[:, None]),
            extrinsics[:, :, None],
        )[..., :3]
        uv = project_camera_space(cam_points, intrinsics[:, :, None])
        valid = (
            (cam_points[..., 2] > 1e-6)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= 1)
        )
        return uv, cam_points[..., 2], valid

    def _sample_patch(self, feature_map: Tensor, uv: Tensor, patch_size: int) -> Tensor:
        b, v, c, h, w = feature_map.shape
        n = uv.shape[2]
        offsets = self._patch_offsets(patch_size, feature_map.device, feature_map.dtype)
        pixel_scale = uv.new_tensor((max(w - 1, 1), max(h - 1, 1)))
        patch_uv = uv[:, :, :, None] + offsets / pixel_scale
        grid = rearrange(patch_uv * 2 - 1, "b v n p xy -> (b v) n p xy")
        sampled = F.grid_sample(
            rearrange(feature_map, "b v c h w -> (b v) c h w"),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return rearrange(sampled, "(b v) c n p -> b v n (p c)", b=b, v=v, n=n)

    @staticmethod
    def _dn_from_knn(anchors: Tensor) -> Tensor:
        b, v, n, _ = anchors.shape
        flat = rearrange(anchors, "b v n xyz -> (b v) n xyz")
        if n <= 1:
            return anchors.new_full((b, v, n), 1e-3)
        k = min(4, n)
        dist = torch.cdist(flat.float(), flat.float())
        knn = dist.topk(k=k, dim=-1, largest=False).values[..., 1:]
        dn = knn.mean(dim=-1).clamp_min(1e-6).to(anchors.dtype)
        return rearrange(dn, "(b v) n -> b v n", b=b, v=v)

    def _reshape_parent_gaussians(self, gaussians: Gaussians, b: int, v: int, n: int) -> Gaussians:
        if gaussians.means.shape[1] != v * n:
            raise ValueError(
                "parent gaussians must match anchors, got "
                f"gaussians={gaussians.means.shape[1]} and anchors={v * n}."
            )
        return Gaussians(
            means=rearrange(gaussians.means, "b (v n) xyz -> b v n xyz", v=v, n=n),
            covariances=rearrange(gaussians.covariances, "b (v n) i j -> b v n i j", v=v, n=n),
            rotations=rearrange(gaussians.rotations, "b (v n) q -> b v n q", v=v, n=n),
            scales=rearrange(gaussians.scales, "b (v n) xyz -> b v n xyz", v=v, n=n),
            harmonics=rearrange(gaussians.harmonics, "b (v n) rgb sh -> b v n rgb sh", v=v, n=n),
            opacities=rearrange(gaussians.opacities, "b (v n) -> b v n", v=v, n=n),
        )

    def forward(
        self,
        anchors: Tensor,
        gs_feature_map: Tensor,
        sr_feature_map: Tensor,
        sr_image: Tensor,
        lr_render_color: Tensor,
        parent_gaussians: Gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> dict[str, Tensor]:
        b, v, n, _ = anchors.shape
        uv, _, valid = self._project(anchors, extrinsics, intrinsics)
        gs_feature = rearrange(gs_feature_map, "b v c h w -> b v (h w) c")
        if gs_feature.shape[2] != n:
            if n % gs_feature.shape[2] != 0:
                raise ValueError(
                    "gs_feature_map must have one feature per LR anchor, or an integer number "
                    f"of anchors per pixel, got feature_count={gs_feature.shape[2]} and anchors={n}."
                )
            gs_feature = gs_feature.repeat_interleave(n // gs_feature.shape[2], dim=2)

        render_stack = torch.cat(
            [
                sr_image,
                lr_render_color,
            ],
            dim=2,
        )
        sampled_condition = self._sample_patch(render_stack, uv, self.cfg.sample_patch_size)
        sampled_condition = rearrange(
            sampled_condition,
            "b v n (p c) -> b v n p c",
            c=6,
        )
        sampled_condition = self.sample_mlp(sampled_condition)

        query = self.gs_norm(self.gs_mlp(gs_feature))
        query_flat = rearrange(query, "b v n c -> (b v n) 1 c")
        kv_flat = rearrange(sampled_condition, "b v n p c -> (b v n) p c")
        refined, _ = self.cross_attn(query_flat, kv_flat, kv_flat, need_weights=False)
        refined = self.post_attn(refined.squeeze(1))
        refined = rearrange(refined, "(b v n) c -> (b v) n c", b=b, v=v, n=n)
        coords_flat = rearrange(anchors, "b v n xyz -> (b v) n xyz")
        refined = self.point_transformer(coords_flat, refined)
        raw_offsets = self.offset_head(refined)
        raw_offsets = rearrange(
            raw_offsets,
            "(b v) n (k xyz) -> b v n k xyz",
            b=b,
            v=v,
            k=self.cfg.num_offsets,
            xyz=3,
        )
        dn = self._dn_from_knn(anchors)
        offsets = torch.tanh(raw_offsets) * dn[..., None, None]
        child_means = anchors[:, :, :, None] + offsets

        if sr_feature_map.shape[-2:] != sr_image.shape[-2:]:
            sr_feature_map = F.interpolate(
                rearrange(sr_feature_map, "b v c h w -> (b v) c h w"),
                size=sr_image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            sr_feature_map = rearrange(sr_feature_map, "(b v) c h w -> b v c h w", b=b, v=v)

        child_sample_map = torch.cat(
            [
                sr_image,
                sr_feature_map,
            ],
            dim=2,
        )
        flat_child_means = rearrange(child_means, "b v n k xyz -> b (v n k) xyz")
        child_uv, _, child_valid = self._project_points_to_views(
            flat_child_means,
            extrinsics,
            intrinsics,
        )
        sampled_child_feature = self._sample_patch(
            child_sample_map,
            child_uv,
            self.cfg.child_sample_patch_size,
        )
        sampled_child_feature = rearrange(
            sampled_child_feature,
            "b view (parent_v n k) pc -> b view parent_v n k pc",
            parent_v=v,
            n=n,
            k=self.cfg.num_offsets,
        )
        child_valid = rearrange(
            child_valid,
            "b view (parent_v n k) -> b view parent_v n k",
            parent_v=v,
            n=n,
            k=self.cfg.num_offsets,
        )
        sampled_child_feature = sampled_child_feature * child_valid[..., None].to(
            sampled_child_feature.dtype
        )
        view_count = sampled_child_feature.shape[1]
        if view_count < self.cfg.child_aggregation_views:
            sampled_child_feature = torch.cat(
                [
                    sampled_child_feature,
                    sampled_child_feature.new_zeros(
                        b,
                        self.cfg.child_aggregation_views - view_count,
                        v,
                        n,
                        self.cfg.num_offsets,
                        sampled_child_feature.shape[-1],
                    ),
                ],
                dim=1,
            )
            child_valid = torch.cat(
                [
                    child_valid,
                    child_valid.new_zeros(
                        b,
                        self.cfg.child_aggregation_views - view_count,
                        v,
                        n,
                        self.cfg.num_offsets,
                    ),
                ],
                dim=1,
            )
        elif view_count > self.cfg.child_aggregation_views:
            sampled_child_feature = sampled_child_feature[:, :self.cfg.child_aggregation_views]
            child_valid = child_valid[:, :self.cfg.child_aggregation_views]
        sampled_child_feature = rearrange(
            sampled_child_feature,
            "b view parent_v n k pc -> b parent_v n k (view pc)",
        )
        child_feature = self.child_feature_fusion(sampled_child_feature)
        child_feature = rearrange(child_feature, "b v n k c -> (b v n) k c")
        for layer in self.child_self_attn:
            child_feature = layer(child_feature)
        child_feature = rearrange(child_feature, "(b v n) k c -> b v n k c", b=b, v=v, n=n)

        raw_attributes = self.attribute_head(child_feature)
        cursor = 0
        raw_scale = raw_attributes[..., cursor:cursor + 3]
        cursor += 3
        raw_opacity = raw_attributes[..., cursor:cursor + 1].squeeze(-1)
        cursor += 1
        raw_rotation = raw_attributes[..., cursor:cursor + 4]
        cursor += 4
        raw_harmonics = raw_attributes[..., cursor:]

        parent_gaussians = self._reshape_parent_gaussians(parent_gaussians, b, v, n)
        parent_scales = parent_gaussians.scales.detach()[:, :, :, None]
        parent_opacities = parent_gaussians.opacities.detach()[:, :, :, None]
        parent_rotations = parent_gaussians.rotations.detach()[:, :, :, None]
        parent_harmonics = parent_gaussians.harmonics.detach()[:, :, :, None]

        base_scales = parent_scales / self.cfg.child_scale_divisor
        base_opacities = parent_opacities / self.cfg.num_offsets
        base_opacities_raw = torch.logit(
            base_opacities.clamp(1e-6, 1 - 1e-6),
            eps=1e-6,
        )

        child_scales = (base_scales + raw_scale).clamp(1e-6, 0.3)
        child_opacities = (base_opacities_raw + raw_opacity).sigmoid()
        child_rotations = F.normalize(parent_rotations + raw_rotation, dim=-1)
        child_harmonics = parent_harmonics + rearrange(
            raw_harmonics,
            "b v n k (rgb sh) -> b v n k rgb sh",
            rgb=3,
        )
        child_covariances = build_covariance(child_scales, child_rotations)
        child_gaussians = Gaussians(
            means=rearrange(child_means, "b v n k xyz -> b (v n k) xyz"),
            covariances=rearrange(child_covariances, "b v n k i j -> b (v n k) i j"),
            rotations=rearrange(child_rotations, "b v n k q -> b (v n k) q"),
            scales=rearrange(child_scales, "b v n k xyz -> b (v n k) xyz"),
            harmonics=rearrange(child_harmonics, "b v n k rgb sh -> b (v n k) rgb sh"),
            opacities=rearrange(child_opacities, "b v n k -> b (v n k)"),
        )
        return {
            "gaussians": child_gaussians,
            "child_means": child_means,
            "offsets": offsets,
            "dn": dn,
            "refined_feature": rearrange(refined, "(b v) n c -> b v n c", b=b, v=v),
            "valid": valid,
            "child_valid": child_valid,
        }
