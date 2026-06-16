from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from .anchor_feature_sampler import AnchorFeatureSampler
from .anchor_geometry_encoder import PointGeometryEncoder
from .anchor_litept_fusion import AnchorLitePTFusion
from ...geometry.projection import (
    homogenize_points,
    project_camera_space,
    transform_cam2world,
    transform_world2cam,
    unproject,
)


@dataclass
class ConditionalDensificationResult:
    child_means: Tensor
    densities: Tensor
    raw_gaussians: Tensor


class ConditionalGaussianDensifier(nn.Module):
    """Aggregate per-view observations and decode children for selected LR anchors."""

    def __init__(
        self,
        gs_feature_dim: int = 256,
        view_feature_dim: int = 128,
        point_feature_dim: int = 8,
        condition_dim: int = 6,
        attention_dim: int = 128,
        attention_heads: int = 8,
        geometry_dim: int = 128,
        litept_token_dim: int = 256,
        litept_path: str = "/space0/mengxl/LitePT-main",
        use_full_litept: bool = True,
        litept_grid_size: float = 0.02,
        num_slots: int = 8,
        slot_dim: int = 32,
        hidden_dim: int = 256,
        raw_gaussian_dim: int = 82,
    ) -> None:
        super().__init__()
        if attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")

        self.attention_heads = attention_heads
        self.head_dim = attention_dim // attention_heads
        self.num_slots = num_slots

        self.feature_sampler = AnchorFeatureSampler(patch_size=1)
        self.query_mlp = nn.Sequential(
            nn.Linear(gs_feature_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
            nn.LayerNorm(attention_dim),
        )
        kv_input_dim = view_feature_dim + point_feature_dim + condition_dim
        self.key_mlp = nn.Sequential(
            nn.LayerNorm(kv_input_dim),
            nn.Linear(kv_input_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.value_mlp = nn.Sequential(
            nn.LayerNorm(kv_input_dim),
            nn.Linear(kv_input_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.condition_bias = nn.Sequential(
            nn.Linear(condition_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, attention_heads),
        )
        self.attention_output = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.LayerNorm(attention_dim),
        )

        self.geometry_encoder = PointGeometryEncoder(
            output_dim=geometry_dim,
        )
        self.litept = AnchorLitePTFusion(
            feature_dim=attention_dim,
            geometry_dim=geometry_dim,
            token_dim=litept_token_dim,
            litept_path=litept_path,
            use_full_litept=use_full_litept,
            litept_grid_size=litept_grid_size,
        )

        self.slot_embeddings = nn.Parameter(torch.empty(num_slots, slot_dim))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)
        offset_input_dim = litept_token_dim + slot_dim
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(offset_input_dim),
            nn.Linear(offset_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.normal_(self.offset_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.offset_mlp[-1].bias)

        residual_input_dim = litept_token_dim + geometry_dim + slot_dim
        self.feature_residual_mlp = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, litept_token_dim),
        )
        gaussian_input_dim = litept_token_dim + geometry_dim + slot_dim
        self.gaussian_mlp = nn.Sequential(
            nn.LayerNorm(gaussian_input_dim),
            nn.Linear(gaussian_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1 + raw_gaussian_dim),
        )
        nn.init.zeros_(self.gaussian_mlp[-1].weight)
        nn.init.zeros_(self.gaussian_mlp[-1].bias)
        with torch.no_grad():
            self.gaussian_mlp[-1].bias[0] = -2.0
            self.gaussian_mlp[-1].bias[7] = 1.0

    @staticmethod
    def _gather_selected(values: Tensor, mask: Tensor) -> Tensor:
        selected_count = mask.sum(dim=1)
        if not torch.equal(
            selected_count,
            selected_count[:1].expand_as(selected_count),
        ):
            raise ValueError(
                "Each batch item must select the same number of LR anchors"
            )
        return torch.stack(
            [sample_values[sample_mask] for sample_values, sample_mask in zip(
                values,
                mask,
                strict=True,
            )],
            dim=0,
        )

    def _aggregate_views(
        self,
        anchors: Tensor,
        gs_features: Tensor,
        view_feature_map: Tensor,
        point_feature_map: Tensor,
        conditions: Tensor,
        projection_valid: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        sampled_view = self.feature_sampler(
            anchors=anchors,
            feature_map=view_feature_map,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        sampled_point = self.feature_sampler(
            anchors=anchors,
            feature_map=point_feature_map,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        valid = sampled_view.valid_mask & sampled_point.valid_mask & projection_valid
        kv_input = torch.cat(
            [sampled_view.features, sampled_point.features, conditions],
            dim=-1,
        )

        query = self.query_mlp(gs_features)
        key = self.key_mlp(kv_input)
        value = self.value_mlp(kv_input)
        query = rearrange(
            query,
            "b n (head d) -> b n head 1 d",
            head=self.attention_heads,
        )
        key = rearrange(
            key,
            "b n v (head d) -> b n head v d",
            head=self.attention_heads,
        )
        value = rearrange(
            value,
            "b n v (head d) -> b n head v d",
            head=self.attention_heads,
        )
        logits = (query * key).sum(dim=-1) * self.head_dim ** -0.5
        logits = logits + rearrange(
            self.condition_bias(conditions),
            "b n v head -> b n head v",
        )
        logits = logits.masked_fill(~valid[:, :, None], -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * valid[:, :, None].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aggregated = (attention[..., None] * value).sum(dim=-2)
        aggregated = rearrange(
            aggregated,
            "b n head d -> b n (head d)",
        )
        return self.attention_output(aggregated)

    def forward(
        self,
        anchors: Tensor,
        gs_features: Tensor,
        view_feature_map: Tensor,
        point_feature_map: Tensor,
        conditions: Tensor,
        projection_valid: Tensor,
        source_view_indices: Tensor,
        densification_mask: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
    ) -> ConditionalDensificationResult:
        selected_anchors = self._gather_selected(anchors, densification_mask)
        selected_gs_features = self._gather_selected(
            gs_features,
            densification_mask,
        )
        aggregated = self._aggregate_views(
            selected_anchors,
            selected_gs_features,
            view_feature_map,
            point_feature_map,
            conditions,
            projection_valid,
            extrinsics,
            intrinsics,
        )
        geometry = self.geometry_encoder(selected_anchors)
        litept_features = self.litept(
            selected_anchors,
            aggregated,
            geometry,
        )

        slots = self.slot_embeddings[None, None].expand(
            selected_anchors.shape[0],
            selected_anchors.shape[1],
            -1,
            -1,
        )
        parent_features = litept_features[:, :, None].expand(
            -1,
            -1,
            self.num_slots,
            -1,
        )
        offset_input = torch.cat([parent_features, slots], dim=-1)
        batch_indices = torch.arange(
            selected_anchors.shape[0],
            device=selected_anchors.device,
        )[:, None]
        source_extrinsics = extrinsics[batch_indices, source_view_indices]
        source_intrinsics = intrinsics[batch_indices, source_view_indices]
        camera_anchors = transform_world2cam(
            homogenize_points(selected_anchors),
            source_extrinsics,
        )[..., :3]
        parent_uv = project_camera_space(
            camera_anchors,
            source_intrinsics,
        )
        lr_height, lr_width = view_feature_map.shape[-2:]
        half_lr_pixel = parent_uv.new_tensor(
            [0.5 / lr_width, 0.5 / lr_height]
        )
        raw_offset = self.offset_mlp(offset_input)
        parent_depth = camera_anchors[..., 2].clamp_min(1e-6)
        fx = source_intrinsics[..., 0, 0].abs().clamp_min(1e-6)
        fy = source_intrinsics[..., 1, 1].abs().clamp_min(1e-6)
        xy_camera_radius = torch.stack(
            [
                half_lr_pixel[0] * parent_depth / fx,
                half_lr_pixel[1] * parent_depth / fy,
            ],
            dim=-1,
        )
        z_camera_radius = parent_depth[..., None] * 0.05
        camera_offset = torch.cat(
            [
                torch.tanh(raw_offset[..., :2])
                * xy_camera_radius[:, :, None],
                torch.tanh(raw_offset[..., 2:3])
                * z_camera_radius[:, :, None],
            ],
            dim=-1,
        )
        child_camera = camera_anchors[:, :, None] + camera_offset

        child_uv = project_camera_space(
            child_camera,
            source_intrinsics[:, :, None],
        )
        clamped_child_uv = parent_uv[:, :, None] + (
            child_uv - parent_uv[:, :, None]
        ).clamp(
            min=-half_lr_pixel,
            max=half_lr_pixel,
        )
        child_camera = unproject(
            clamped_child_uv,
            child_camera[..., 2],
            source_intrinsics[:, :, None],
        )
        child_means = transform_cam2world(
            homogenize_points(child_camera),
            source_extrinsics[:, :, None],
        )[..., :3]

        child_geometry = self.geometry_encoder(
            rearrange(child_means, "b n k xyz -> b (n k) xyz")
        )
        child_geometry = rearrange(
            child_geometry,
            "b (n k) c -> b n k c",
            n=selected_anchors.shape[1],
            k=self.num_slots,
        )
        residual_input = torch.cat(
            [parent_features, child_geometry, slots],
            dim=-1,
        )
        updated_features = parent_features + self.feature_residual_mlp(
            residual_input
        )
        raw_parameters = self.gaussian_mlp(
            torch.cat([updated_features, child_geometry, slots], dim=-1)
        )
        return ConditionalDensificationResult(
            child_means=child_means,
            densities=raw_parameters[..., 0].sigmoid(),
            raw_gaussians=raw_parameters[..., 1:],
        )
