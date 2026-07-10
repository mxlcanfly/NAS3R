from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

import pointops


@dataclass
class ReSplatPointTransformerCfg:
    enabled: bool = True
    channels: int = 512
    knn_samples: int = 16
    num_blocks: int = 4
    attn_proj_channels: int = 64
    input_dim: int | None = None


class KNNAttention(nn.Module):
    """ReSplat-style KNN attention without multi-view attention."""

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        proj_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.knn_samples = knn_samples
        self.proj_channels = proj_channels

        qkv_channels = proj_channels or channels
        self.qkv = nn.Linear(channels, qkv_channels * 3, bias=False)
        self.proj = nn.Linear(qkv_channels, channels)

    def forward(self, pxo: tuple[Tensor, Tensor, Tensor], knn_idx: Tensor | None = None) -> Tensor:
        points, features, offsets = pxo
        channels = features.size(1)
        if self.proj_channels is not None:
            channels = self.proj_channels
        head_dim = channels
        scale = head_dim ** -0.5

        qkv = self.qkv(features)
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

        key_neighbors, idx = pointops.knn_query_and_group(
            key.contiguous(),
            points,
            offsets,
            new_xyz=points,
            new_offset=offsets,
            idx=knn_idx,
            nsample=self.knn_samples,
            with_xyz=False,
        )
        value_neighbors, _ = pointops.knn_query_and_group(
            value.contiguous(),
            points,
            offsets,
            new_xyz=points,
            new_offset=offsets,
            idx=idx,
            nsample=self.knn_samples,
            with_xyz=False,
        )

        scores = torch.matmul(query.unsqueeze(1), key_neighbors.permute(0, 2, 1)) * scale
        out = torch.matmul(torch.softmax(scores, dim=2), value_neighbors).squeeze(1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        attn_proj_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = KNNAttention(
            channels,
            knn_samples=knn_samples,
            proj_channels=attn_proj_channels,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, pxo: tuple[Tensor, Tensor, Tensor], knn_idx: Tensor | None = None) -> Tensor:
        points, features, offsets = pxo
        features = features + self.attn((points, self.norm1(features), offsets), knn_idx=knn_idx)
        features = features + self.mlp(self.norm2(features))
        return features


class PlainPointTransformer(nn.Module):
    """Lightweight ReSplat PointTransformer with mv-attention disabled."""

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        num_blocks: int = 4,
        attn_proj_channels: int | None = None,
        cache_knn_idx: bool = True,
    ) -> None:
        super().__init__()
        self.cache_knn_idx = cache_knn_idx
        self.knn_samples = knn_samples
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    knn_samples=knn_samples,
                    attn_proj_channels=attn_proj_channels,
                )
                for _ in range(num_blocks)
            ]
        )

    def compute_knn(self, points: Tensor, offsets: Tensor) -> Tensor:
        knn_idx, _ = pointops.knn_query(
            self.knn_samples,
            points.contiguous(),
            offsets,
            points.contiguous(),
            offsets,
        )
        return knn_idx

    def forward(
        self,
        pxo: tuple[Tensor, Tensor, Tensor],
        cached_knn_idx: Tensor | None = None,
        return_knn_idx: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        points, features, offsets = pxo
        knn_idx = cached_knn_idx
        if knn_idx is None and self.cache_knn_idx:
            knn_idx = self.compute_knn(points, offsets)

        for block in self.blocks:
            features = block((points, features, offsets), knn_idx=knn_idx)

        if return_knn_idx:
            if knn_idx is None:
                knn_idx = self.compute_knn(points, offsets)
            return features, knn_idx
        return features


class PointLinearWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, pxo: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        points, features, offsets = pxo
        return points, self.linear(features), offsets


class ReSplatGaussianPointTransformer(nn.Module):
    """Projection wrapper for later Gaussian/anchor feature refinement."""

    def __init__(self, cfg: ReSplatPointTransformerCfg, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim or input_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, cfg.channels),
        )
        self.point_transformer = PlainPointTransformer(
            cfg.channels,
            knn_samples=cfg.knn_samples,
            num_blocks=cfg.num_blocks,
            attn_proj_channels=cfg.attn_proj_channels,
        )

    @staticmethod
    def batch_offsets(batch_size: int, points_per_batch: int, device: torch.device) -> Tensor:
        return torch.arange(1, batch_size + 1, device=device, dtype=torch.long) * points_per_batch

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        offsets: Tensor | None = None,
        return_knn_idx: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if points.shape[:2] != features.shape[:2]:
            raise ValueError(
                "points and features must share batch/point axes, got "
                f"{tuple(points.shape)} and {tuple(features.shape)}"
            )
        b, n, _ = points.shape
        if offsets is None:
            offsets = self.batch_offsets(b, n, points.device)

        flat_points = points.reshape(b * n, 3)
        flat_features = self.input_proj(features.reshape(b * n, features.shape[-1]))
        return self.point_transformer(
            (flat_points, flat_features, offsets),
            return_knn_idx=return_knn_idx,
        )
