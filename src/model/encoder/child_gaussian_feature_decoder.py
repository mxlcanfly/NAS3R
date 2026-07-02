from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ..types import Gaussians
from .common.gaussians import build_covariance
from ...geometry.projection import homogenize_points, project_camera_space, transform_world2cam


@dataclass
class ChildGaussianFeatureDecoderCfg:
    enabled: bool = True
    sr_feature_dim: int = 256
    hidden_dim: int = 256
    self_attn_layers: int = 4
    self_attn_heads: int = 8
    group_norm_groups: int = 16
    center_fourier_frequencies: int = 6
    center_fourier_scale: float = 1.0
    child_scale_divisor: float = 2.0
    clamp_min_scale: float = 1e-6
    clamp_max_scale: float = 0.3


class ChildGaussianFeatureDecoder(nn.Module):
    def __init__(self, cfg: ChildGaussianFeatureDecoderCfg, d_sh: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_sh = d_sh
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(cfg.sr_feature_dim),
            nn.Linear(cfg.sr_feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.center_fourier_dim = 3 * 2 * cfg.center_fourier_frequencies
        self.center_proj = nn.Sequential(
            nn.LayerNorm(self.center_fourier_dim),
            nn.Linear(self.center_fourier_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.feature_center_fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.self_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_dim,
                    nhead=cfg.self_attn_heads,
                    dim_feedforward=cfg.hidden_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.self_attn_layers)
            ]
        )
        groups = min(cfg.group_norm_groups, cfg.hidden_dim)
        while cfg.hidden_dim % groups != 0:
            groups -= 1
        self.group_norm = nn.GroupNorm(groups, cfg.hidden_dim)
        self.attribute_dim = 3 + 1 + 4 + 3 * d_sh
        self.attribute_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, self.attribute_dim),
        )
        self._init_linear(self.feature_proj)
        self._init_linear(self.center_proj)
        self._init_linear(self.feature_center_fuse)
        self._init_attribute_head()

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _init_attribute_head(self) -> None:
        for layer in self.attribute_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.attribute_head[-1].weight)
        nn.init.zeros_(self.attribute_head[-1].bias)

    @staticmethod
    def _reshape_parent_gaussians(parent_gaussians: Gaussians, b: int, v: int, n: int) -> Gaussians:
        if parent_gaussians.means.shape[1] != v * n:
            raise ValueError(
                "parent gaussians must match [B, V*N], got "
                f"{tuple(parent_gaussians.means.shape)} and V*N={v * n}."
            )
        return Gaussians(
            means=rearrange(parent_gaussians.means, "b (v n) xyz -> b v n xyz", v=v, n=n),
            covariances=rearrange(parent_gaussians.covariances, "b (v n) i j -> b v n i j", v=v, n=n),
            rotations=rearrange(parent_gaussians.rotations, "b (v n) q -> b v n q", v=v, n=n),
            scales=rearrange(parent_gaussians.scales, "b (v n) xyz -> b v n xyz", v=v, n=n),
            harmonics=rearrange(parent_gaussians.harmonics, "b (v n) rgb sh -> b v n rgb sh", v=v, n=n),
            opacities=rearrange(parent_gaussians.opacities, "b (v n) -> b v n", v=v, n=n),
        )

    @staticmethod
    def _project_points(points: Tensor, extrinsics: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor]:
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
        return uv, valid

    @staticmethod
    def _sample_point_features(feature_map: Tensor, uv: Tensor) -> Tensor:
        b, view_count, c, _, _ = feature_map.shape
        point_count = uv.shape[2]
        grid = rearrange(uv * 2 - 1, "b v n xy -> (b v) n 1 xy")
        sampled = F.grid_sample(
            rearrange(feature_map, "b v c h w -> (b v) c h w"),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return rearrange(sampled, "(b v) c n 1 -> b v n c", b=b, v=view_count, n=point_count)

    @staticmethod
    def _split_attributes(raw_attributes: Tensor, d_sh: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cursor = 0
        raw_scale = raw_attributes[..., cursor: cursor + 3]
        cursor += 3
        raw_opacity = raw_attributes[..., cursor: cursor + 1].squeeze(-1)
        cursor += 1
        raw_rotation = raw_attributes[..., cursor: cursor + 4]
        cursor += 4
        raw_harmonics = rearrange(raw_attributes[..., cursor:], "... (rgb sh) -> ... rgb sh", rgb=3, sh=d_sh)
        return raw_scale, raw_opacity, raw_rotation, raw_harmonics

    def _encode_centers(self, child_centers: Tensor) -> Tensor:
        if self.cfg.center_fourier_frequencies <= 0:
            raise ValueError("center_fourier_frequencies must be positive.")
        scaled_centers = child_centers * self.cfg.center_fourier_scale
        frequencies = (2.0 ** torch.arange(
            self.cfg.center_fourier_frequencies,
            device=child_centers.device,
            dtype=child_centers.dtype,
        ))
        encoded = scaled_centers[..., None] * frequencies
        encoded = torch.cat([encoded.sin(), encoded.cos()], dim=-1)
        return rearrange(encoded, "b v n k xyz f -> b v n k (xyz f)")

    def forward(
        self,
        child_centers: Tensor,
        sr_feature_map: Tensor,
        parent_gaussians: Gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> dict[str, Tensor | Gaussians]:
        b, v, n, k, _ = child_centers.shape
        flat_child_centers = rearrange(child_centers, "b v n k xyz -> b (v n k) xyz")
        uv, valid = self._project_points(flat_child_centers, extrinsics, intrinsics)
        sampled = self._sample_point_features(sr_feature_map, uv)
        valid_weight = valid.to(sampled.dtype)
        child_features = (sampled * valid_weight[..., None]).sum(dim=1)
        child_features = child_features / valid_weight.sum(dim=1, keepdim=False).clamp_min(1.0)[..., None]
        child_features = rearrange(child_features, "b (v n k) c -> b v n k c", v=v, n=n, k=k)

        child_features = self.feature_proj(child_features)
        parents = self._reshape_parent_gaussians(parent_gaussians, b, v, n)
        center_features = self.center_proj(self._encode_centers(child_centers))
        child_features = self.feature_center_fuse(torch.cat([child_features, center_features], dim=-1))
        child_features = rearrange(child_features, "b v n k c -> (b v n) k c")
        for layer in self.self_attn:
            child_features = layer(child_features)
        child_features = rearrange(child_features, "(b v n) k c -> b v n k c", b=b, v=v, n=n)
        child_features = rearrange(child_features, "b v n k c -> (b v n k) c")
        child_features = self.group_norm(child_features[:, :, None]).squeeze(-1)
        child_features = rearrange(child_features, "(b v n k) c -> b v n k c", b=b, v=v, n=n, k=k)

        raw_scale, raw_opacity, raw_rotation, raw_harmonics = self._split_attributes(
            self.attribute_head(child_features),
            self.d_sh,
        )
        parent_scales = parents.scales.detach()[:, :, :, None]
        parent_opacities = parents.opacities.detach()[:, :, :, None]
        parent_rotations = parents.rotations.detach()[:, :, :, None]
        parent_harmonics = parents.harmonics.detach()[:, :, :, None]

        base_scales = parent_scales / self.cfg.child_scale_divisor
        base_opacities = 1.0 - (1.0 - parent_opacities.clamp(0.0, 1.0)).pow(1.0 / k)
        base_opacity_logits = torch.logit(base_opacities.clamp(1e-6, 1.0 - 1e-6))

        child_scales = (base_scales + raw_scale).clamp(
            min=self.cfg.clamp_min_scale,
            max=self.cfg.clamp_max_scale,
        )
        child_rotations = F.normalize(
            parent_rotations + raw_rotation,
            dim=-1,
            eps=1e-8,
        )
        child_harmonics = parent_harmonics + raw_harmonics
        child_opacities = (base_opacity_logits + raw_opacity).sigmoid()
        child_covariances = build_covariance(child_scales, child_rotations)

        child_gaussians = Gaussians(
            means=rearrange(child_centers, "b v n k xyz -> b (v n k) xyz"),
            covariances=rearrange(child_covariances, "b v n k i j -> b (v n k) i j"),
            rotations=rearrange(child_rotations, "b v n k q -> b (v n k) q"),
            scales=rearrange(child_scales, "b v n k xyz -> b (v n k) xyz"),
            harmonics=rearrange(child_harmonics, "b v n k rgb sh -> b (v n k) rgb sh"),
            opacities=rearrange(child_opacities, "b v n k -> b (v n k)"),
        )
        return {
            "gaussians": child_gaussians,
            "features": child_features,
            "valid": rearrange(valid, "b view (v n k) -> b view v n k", v=v, n=n, k=k),
        }
