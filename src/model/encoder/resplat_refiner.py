"""Single-step Gaussian refinement from an exact observation cue.

This module intentionally keeps the refinement independent from the frozen
NAS3R-M encoder. ReSplat-style feature and RGB rendering residuals are lifted
to Gaussians by the renderer before this module is called. The refiner uses
Gaussian parameters plus one physically aligned multi-channel cue per
Gaussian, local 3D kNN attention, and a four-layer Gaussian residual head.
"""

from __future__ import annotations

import warnings

import pointops
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from ..types import Gaussians
from .common.gaussians import build_covariance


class ResNet18Features(nn.Module):
    """Frozen ResNet-18 feature pyramid used by ReSplat."""

    def __init__(self) -> None:
        super().__init__()
        try:
            network = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            warnings.warn(
                "Could not load ImageNet ResNet-18 weights for render-error "
                f"features ({exc}); using random frozen weights."
            )
            network = resnet18(weights=None)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, image: Tensor) -> list[Tensor]:
        features = []
        image = self.conv1(image)
        features.append(image)
        image = self.relu(self.bn1(image))
        image = self.maxpool(image)
        image = self.layer1(image)
        features.append(image)
        image = self.layer2(image)
        features.append(image)
        return features


class KNNAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int = 8,
        projection_channels: int = 64,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.projection_channels = projection_channels
        self.qkv = nn.Linear(channels, projection_channels * 3, bias=False)
        self.proj = nn.Linear(projection_channels, channels)

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        offsets: Tensor,
        knn_index: Tensor,
        query_chunk_size: int | None = None,
    ) -> Tensor:
        query, key, value = self.qkv(features).chunk(3, dim=-1)
        chunk_size = query.shape[0] if query_chunk_size is None else query_chunk_size
        if chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {chunk_size}")

        # pointops uses -1 as a padded neighbor. Append one zero row so direct
        # indexing has exactly the same semantics while allowing query chunks
        # to access keys/values from the complete scene.
        key_with_padding = torch.cat((key, key.new_zeros(1, key.shape[-1])), dim=0)
        value_with_padding = torch.cat(
            (value, value.new_zeros(1, value.shape[-1])), dim=0
        )
        outputs = []
        for start in range(0, query.shape[0], chunk_size):
            end = min(start + chunk_size, query.shape[0])
            chunk_index = knn_index[start:end].long()
            grouped_key = key_with_padding[chunk_index]
            grouped_value = value_with_padding[chunk_index]
            chunk_query = query[start:end]
            scores = (chunk_query.unsqueeze(1) * grouped_key).sum(dim=-1)
            scores = scores * chunk_query.shape[-1] ** -0.5
            outputs.append(
                self.proj(
                    (
                        scores.softmax(dim=-1).unsqueeze(-1) * grouped_value
                    ).sum(dim=1)
                )
            )
        return torch.cat(outputs, dim=0)


class PointTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        projection_channels: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = KNNAttention(channels, samples, projection_channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        offsets: Tensor,
        knn_index: Tensor,
        query_chunk_size: int | None = None,
    ) -> Tensor:
        features = features + self.attn(
            points,
            self.norm1(features),
            offsets,
            knn_index,
            query_chunk_size,
        )
        return features + self.mlp(self.norm2(features))


class PlainPointTransformer(nn.Module):
    """ReSplat PlainPointTransformer with one cached 3D kNN graph."""

    def __init__(
        self,
        channels: int,
        samples: int,
        blocks: int,
        projection_channels: int,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.query_chunk_size = query_chunk_size
        self.blocks = nn.ModuleList(
            [
                PointTransformerBlock(
                    channels,
                    samples,
                    projection_channels,
                )
                for _ in range(blocks)
            ]
        )

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        offsets: Tensor,
        knn_index: Tensor | None = None,
    ) -> Tensor:
        # Identical to ReSplat's cache_knn_idx=True path: the neighbor graph is
        # built from Gaussian means once and shared by every transformer block.
        if knn_index is None:
            knn_index, _ = pointops.knn_query(
                self.samples,
                points,
                offsets,
                points,
                offsets,
            )
        for block in self.blocks:
            features = block(
                points,
                features,
                offsets,
                knn_index,
                self.query_chunk_size,
            )
        return features


class ReSplatSingleRefiner(nn.Module):
    """One same-point-count refinement step for NAS3R-M Gaussians."""

    def __init__(
        self,
        sh_degree: int,
        channels: int = 512,
        knn_samples: int = 8,
        point_blocks: int = 4,
        knn_projection_channels: int = 64,
        scale_min: float = 1e-6,
        feature_error_dim: int = 256,
        pt_chunk_size: int | None = 32768,
    ) -> None:
        super().__init__()
        self.sh_dim = 3 * (sh_degree + 1) ** 2
        self.gaussian_dim = 3 + 3 + 4 + 1 + self.sh_dim
        self.channels = channels
        self.knn_samples = knn_samples
        self.scale_min = scale_min

        # Keep the complete frozen ResNet-18 pyramid error (64+64+128) after
        # renderer-side lifting. RGB stays a separate three-channel cue until
        # the Gaussian-side trainable projection.
        self.feature_error_dim = feature_error_dim
        self.observation_cue_dim = self.feature_error_dim
        self.feature_extractor = ResNet18Features()
        self.rgb_error_proj = nn.Sequential(
            nn.Linear(3, self.feature_error_dim),
            nn.LayerNorm(self.feature_error_dim),
        )
        # Preserve the physical zero point: identical render/GT must yield a
        # zero RGB projection rather than a bias-induced refinement cue.
        nn.init.zeros_(self.rgb_error_proj[0].bias)
        self.input_proj = nn.Linear(
            self.gaussian_dim + self.observation_cue_dim,
            channels,
        )
        self.update_module = PlainPointTransformer(
            channels,
            knn_samples,
            point_blocks,
            knn_projection_channels,
            query_chunk_size=pt_chunk_size,
        )
        self.update_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, self.gaussian_dim),
        )
        nn.init.zeros_(self.update_head[-1].weight)
        nn.init.zeros_(self.update_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    @staticmethod
    def _normalize_imagenet(image: Tensor) -> Tensor:
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (image - mean) / std

    def build_pixel_render_error(
        self,
        rendered: Tensor,
        ground_truth: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build full signed feature and RGB errors on the context image grid.

        The sign follows ReSplat exactly: rendering minus observation. Frozen
        ResNet features are aligned to render resolution before differencing.
        No channel compression or learned mapping occurs before lifting.
        """
        if rendered.shape != ground_truth.shape or rendered.ndim != 5:
            raise ValueError(
                "rendered and ground_truth must share [B,V,3,H,W], got "
                f"{tuple(rendered.shape)} and {tuple(ground_truth.shape)}"
            )
        batch, views, _, height, width = rendered.shape
        rendered_flat = rearrange(rendered.detach(), "b v c h w -> (b v) c h w")
        ground_truth_flat = rearrange(
            ground_truth.detach(),
            "b v c h w -> (b v) c h w",
        )
        both = torch.cat((rendered_flat, ground_truth_flat), dim=0)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=rendered.is_cuda,
        ):
            pyramid = self.feature_extractor(self._normalize_imagenet(both))
            aligned = [
                F.interpolate(
                    feature,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                for feature in pyramid
            ]
            feature_error = torch.cat(
                [
                    (
                        feature[: batch * views]
                        - feature[batch * views :]
                    ).float()
                    for feature in aligned
                ],
                dim=1,
            )

        feature_error = rearrange(
            feature_error,
            "(b v) c h w -> b v c h w",
            b=batch,
            v=views,
        )
        rgb_error = rendered.detach().float() - ground_truth.detach().float()
        if feature_error.shape[2] != self.feature_error_dim:
            raise RuntimeError(
                f"Expected {self.feature_error_dim} ResNet error channels, got "
                f"{feature_error.shape[2]}"
            )
        return feature_error, rgb_error

    def fuse_observation_cue(
        self,
        gaussian_feature_error: Tensor,
        gaussian_rgb_error: Tensor,
    ) -> Tensor:
        if gaussian_feature_error.shape[-1] != self.feature_error_dim:
            raise ValueError(
                f"Gaussian feature error must have {self.feature_error_dim} "
                f"channels, got {gaussian_feature_error.shape[-1]}"
            )
        if gaussian_rgb_error.shape[:-1] != gaussian_feature_error.shape[:-1]:
            raise ValueError("Feature and RGB Gaussian cues must align")
        if gaussian_rgb_error.shape[-1] != 3:
            raise ValueError("Gaussian RGB error must have three channels")
        return gaussian_feature_error + self.rgb_error_proj(gaussian_rgb_error)

    def _pack_gaussians(self, gaussians: Gaussians) -> Tensor:
        opacity_raw = torch.logit(
            gaussians.opacities.detach().clamp(1e-6, 1 - 1e-6)
        )[..., None]
        harmonics = rearrange(
            gaussians.harmonics.detach(),
            "b n rgb sh -> b n (rgb sh)",
        )
        return torch.cat(
            (
                gaussians.means.detach(),
                gaussians.scales.detach(),
                gaussians.rotations.detach(),
                opacity_raw,
                harmonics,
            ),
            dim=-1,
        )

    def forward(
        self,
        gaussians: Gaussians,
        gaussian_feature_error: Tensor,
        gaussian_rgb_error: Tensor,
    ) -> tuple[Gaussians, dict[str, Tensor]]:
        batch, points_per_batch = gaussians.means.shape[:2]
        expected_feature_shape = (batch, points_per_batch, self.feature_error_dim)
        expected_rgb_shape = (batch, points_per_batch, 3)
        if tuple(gaussian_feature_error.shape) != expected_feature_shape:
            raise ValueError(
                "gaussian_feature_error must align one-to-one with Gaussians: "
                f"expected={expected_feature_shape}, "
                f"got={tuple(gaussian_feature_error.shape)}"
            )
        if tuple(gaussian_rgb_error.shape) != expected_rgb_shape:
            raise ValueError(
                "gaussian_rgb_error must align one-to-one with Gaussians: "
                f"expected={expected_rgb_shape}, got={tuple(gaussian_rgb_error.shape)}"
            )

        points = rearrange(
            gaussians.means.detach().float(),
            "b n c -> (b n) c",
        ).contiguous()
        offsets = torch.arange(
            1,
            batch + 1,
            device=points.device,
            dtype=torch.int32,
        ) * points_per_batch
        # Run the point transformer and update head in bfloat16. Positions stay
        # float32 for the 3D kNN graph.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=points.is_cuda,
        ):
            packed = self._pack_gaussians(gaussians)
            render_error = self.fuse_observation_cue(
                gaussian_feature_error.detach(),
                gaussian_rgb_error.detach(),
            )
            features = self.input_proj(
                torch.cat((packed, render_error), dim=-1)
            )
            features = rearrange(features, "b n c -> (b n) c")
            features = self.update_module(points, features, offsets)
            delta = self.update_head(features)

        delta = rearrange(delta.float(), "(b n) c -> b n c", b=batch)
        delta_mean, delta_scale, delta_rotation, delta_opacity, delta_sh = delta.split(
            (3, 3, 4, 1, self.sh_dim),
            dim=-1,
        )

        parent_means = gaussians.means.detach()
        parent_scales = gaussians.scales.detach()
        parent_rotations = gaussians.rotations.detach()
        parent_opacity_raw = torch.logit(
            gaussians.opacities.detach().clamp(1e-6, 1 - 1e-6)
        )[..., None]
        parent_sh = rearrange(
            gaussians.harmonics.detach(),
            "b n rgb sh -> b n (rgb sh)",
        )

        means = parent_means + delta_mean
        scales = (parent_scales + delta_scale).clamp_min(self.scale_min)
        rotations_raw = parent_rotations + delta_rotation
        rotations = F.normalize(rotations_raw, dim=-1, eps=1e-8)
        opacities = torch.sigmoid(parent_opacity_raw + delta_opacity).squeeze(-1)
        harmonics = rearrange(
            parent_sh + delta_sh,
            "b n (rgb sh) -> b n rgb sh",
            rgb=3,
        )
        covariances = build_covariance(scales, rotations)

        refined = Gaussians(
            means=means,
            covariances=covariances,
            rotations=rotations,
            scales=scales,
            harmonics=harmonics,
            opacities=opacities,
        )
        diagnostics = {
            "delta_mean": delta_mean,
            "delta_scale": delta_scale,
            "delta_rotation": delta_rotation,
            "delta_opacity": delta_opacity,
            "delta_sh": delta_sh,
            "gaussian_feature_error": gaussian_feature_error.detach(),
            "gaussian_rgb_error": gaussian_rgb_error.detach(),
            "render_error_cue": render_error.detach(),
        }
        return refined, diagnostics
