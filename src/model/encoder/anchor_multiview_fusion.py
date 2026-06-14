from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .anchor_feature_sampler import AnchorFeatureSampler
from .anchor_feature_unet import AnchorFeatureUNet
from .anchor_geometry_encoder import PointGeometryEncoder


@dataclass
class AnchorMultiViewFusionResult:
    child_means: Tensor
    densities: Tensor
    raw_gaussians: Tensor


class AnchorMultiViewFeatureFusion(nn.Module):
    """Fuse projected view features for LR Gaussians using soft visibility priors."""

    def __init__(
        self,
        feature_dim: int = 128,
        geometry_dim: int = 128,
        geometry_num_frequencies: int = 6,
        hidden_dim: int = 128,
        occlusion_tau: float = 10.0,
        num_slots: int = 8,
        slot_dim: int = 32,
        use_feature_unet: bool = True,
        unet_base_dim: int = 64,
        unet_attention_blocks: int = 4,
        unet_num_heads: int = 4,
        offset_hidden_dim: int = 256,
        hr_feature_dim: int = 32,
        raw_gaussian_dim: int = 82,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if occlusion_tau <= 0:
            raise ValueError("occlusion_tau must be positive.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive.")
        self.occlusion_tau = occlusion_tau
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.slot_embeddings = nn.Parameter(torch.empty(num_slots, slot_dim))
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)
        slot_feature_dim = geometry_dim + feature_dim + slot_dim
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(slot_feature_dim),
            nn.Linear(slot_feature_dim, offset_hidden_dim),
            nn.GELU(),
            nn.Linear(offset_hidden_dim, offset_hidden_dim),
            nn.GELU(),
            nn.Linear(offset_hidden_dim, 3),
        )
        nn.init.normal_(self.offset_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.offset_mlp[-1].bias)
        anchor_feature_dim = geometry_dim + feature_dim
        self.hr_feature_sampler = AnchorFeatureSampler(patch_size=1)
        self.hr_residual_proj = nn.Sequential(
            nn.LayerNorm(hr_feature_dim),
            nn.Linear(hr_feature_dim, anchor_feature_dim),
            nn.GELU(),
            nn.Linear(anchor_feature_dim, anchor_feature_dim),
        )
        self.gaussian_mlp = nn.Sequential(
            nn.LayerNorm(slot_feature_dim),
            nn.Linear(slot_feature_dim, offset_hidden_dim),
            nn.GELU(),
            nn.Linear(offset_hidden_dim, offset_hidden_dim),
            nn.GELU(),
            nn.Linear(offset_hidden_dim, 1 + raw_gaussian_dim),
        )
        nn.init.zeros_(self.gaussian_mlp[-1].weight)
        nn.init.zeros_(self.gaussian_mlp[-1].bias)
        with torch.no_grad():
            # Layout: density | scale(3) | rotation_xyzw(4) | SH.
            self.gaussian_mlp[-1].bias[0] = -2.0
            self.gaussian_mlp[-1].bias[7] = 1.0
        self.feature_unet = (
            AnchorFeatureUNet(
                feature_dim=geometry_dim + feature_dim,
                base_dim=unet_base_dim,
                num_attention_blocks=unet_attention_blocks,
                num_heads=unet_num_heads,
            )
            if use_feature_unet
            else None
        )
        self.feature_sampler = AnchorFeatureSampler(patch_size=1)
        self.geometry_encoder = PointGeometryEncoder(
            num_frequencies=geometry_num_frequencies,
            output_dim=geometry_dim,
            hidden_dim=hidden_dim,
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(geometry_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        anchors: Tensor,
        feature_map: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        source_view_indices: Tensor,
        occlusion_delta: Tensor,
        occlusion_valid_mask: Tensor,
        transmittance: Tensor,
        transmittance_valid_mask: Tensor,
        consistency_weight: Tensor,
        consistency_valid_mask: Tensor,
        anchor_grid_shape: tuple[int, int, int, int, int],
        hr_feature_map: Tensor,
    ) -> AnchorMultiViewFusionResult:
        samples = self.feature_sampler(
            anchors=anchors,
            feature_map=feature_map,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            source_view=source_view_indices,
        )
        geometry_query = self.geometry_encoder(anchors)
        geometry_per_view = geometry_query[:, :, None].expand(
            -1,
            -1,
            samples.features.shape[2],
            -1,
        )
        base_logit = self.score_mlp(
            torch.cat([geometry_per_view, samples.features], dim=-1)
        ).squeeze(-1)

        # Positive delta means the anchor lies behind the sampled surface.
        raw_occlusion_weight = torch.sigmoid(
            -self.occlusion_tau * occlusion_delta
        )
        occlusion_weight = torch.where(
            occlusion_valid_mask,
            raw_occlusion_weight,
            torch.ones_like(raw_occlusion_weight),
        )
        transmittance_weight = torch.where(
            transmittance_valid_mask,
            transmittance.clamp(0, 1),
            torch.ones_like(transmittance),
        )
        physical_weight = (
            occlusion_weight
            + (1 - occlusion_weight) * transmittance_weight
        )
        similarity_weight = torch.where(
            consistency_valid_mask,
            consistency_weight.clamp(0, 1),
            torch.ones_like(consistency_weight),
        )
        joint_weight = physical_weight * similarity_weight

        valid_mask = samples.valid_mask & consistency_valid_mask
        source_mask = torch.nn.functional.one_hot(
            source_view_indices.long(),
            num_classes=samples.features.shape[2],
        ).bool()
        valid_mask = valid_mask | source_mask

        logits = base_logit + torch.log(joint_weight.clamp_min(self.epsilon))
        logits = logits.masked_fill(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
        )
        attention_weight = torch.softmax(logits, dim=-1)
        fused_visual_feature = (
            attention_weight[..., None] * samples.features
        ).sum(dim=2)
        fused_feature = torch.cat(
            [geometry_query, fused_visual_feature],
            dim=-1,
        )
        if self.feature_unet is not None:
            fused_feature = self.feature_unet(
                fused_feature,
                *anchor_grid_shape,
            )
        slot_features = torch.cat(
            [
                fused_feature[:, :, None].expand(
                    -1,
                    -1,
                    self.num_slots,
                    -1,
                ),
                self.slot_embeddings[None, None].expand(
                    fused_feature.shape[0],
                    fused_feature.shape[1],
                    -1,
                    -1,
                ),
            ],
            dim=-1,
        )
        offsets = self.offset_mlp(slot_features)
        child_means = anchors[:, :, None].detach() + offsets

        batch_size, num_anchors, num_slots = child_means.shape[:3]
        child_source_views = source_view_indices[:, :, None].expand(
            -1,
            -1,
            num_slots,
        )
        child_samples = self.hr_feature_sampler(
            anchors=child_means.reshape(batch_size, -1, 3),
            feature_map=hr_feature_map,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        source_indices = child_source_views.reshape(
            batch_size,
            -1,
            1,
            1,
        ).expand(-1, -1, 1, child_samples.features.shape[-1])
        source_hr_feature = child_samples.features.gather(
            dim=2,
            index=source_indices,
        ).squeeze(2)
        source_valid = child_samples.valid_mask.gather(
            dim=2,
            index=child_source_views.reshape(batch_size, -1, 1),
        ).squeeze(2)
        source_hr_feature = source_hr_feature.reshape(
            batch_size,
            num_anchors,
            num_slots,
            -1,
        )
        source_valid = source_valid.reshape(
            batch_size,
            num_anchors,
            num_slots,
        )
        hr_residual = self.hr_residual_proj(source_hr_feature)
        hr_residual = torch.where(
            source_valid[..., None],
            hr_residual,
            torch.zeros_like(hr_residual),
        )

        refined_anchor_feature = fused_feature[:, :, None] + hr_residual
        gaussian_decoder_input = torch.cat(
            [
                refined_anchor_feature,
                self.slot_embeddings[None, None].expand(
                    batch_size,
                    num_anchors,
                    -1,
                    -1,
                ),
            ],
            dim=-1,
        )
        raw_parameters = self.gaussian_mlp(gaussian_decoder_input)
        densities = raw_parameters[..., 0].sigmoid()
        raw_gaussians = raw_parameters[..., 1:]

        return AnchorMultiViewFusionResult(
            child_means=child_means,
            densities=densities,
            raw_gaussians=raw_gaussians,
        )
