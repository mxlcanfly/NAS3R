import math
from dataclasses import dataclass

import torch
from torch import nn

try:
    from pytorch3d.ops import knn_points
except ImportError:
    knn_points = None


@dataclass
class AnchorGeometryEncoding:
    query: torch.Tensor
    spacing: torch.Tensor


class AnchorGeometryEncoder(nn.Module):
    """Build Anchor3DGS-style geometric queries from LR Gaussian centers."""

    def __init__(
        self,
        num_frequencies: int = 6,
        knn: int = 8,
        query_dim: int = 128,
        hidden_dim: int = 256,
        knn_chunk_size: int = 256,
        spacing_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive.")
        if knn <= 0:
            raise ValueError("knn must be positive.")
        if knn_chunk_size <= 0:
            raise ValueError("knn_chunk_size must be positive.")

        self.num_frequencies = num_frequencies
        self.knn = knn
        self.knn_chunk_size = knn_chunk_size
        self.spacing_eps = spacing_eps

        position_dim = 3 + 3 * 2 * num_frequencies
        self.query_mlp = nn.Sequential(
            nn.Linear(position_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
            nn.LayerNorm(query_dim),
        )

    def _fourier_encode(self, anchors: torch.Tensor) -> torch.Tensor:
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        angles = anchors[..., None] * frequencies * math.pi
        periodic = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.cat((anchors, periodic.flatten(start_dim=-2)), dim=-1)

    def _local_spacing(self, anchors: torch.Tensor) -> torch.Tensor:
        num_anchors = anchors.shape[1]
        if num_anchors <= self.knn:
            raise ValueError(
                f"Need more than {self.knn} anchors to compute KNN spacing, "
                f"got {num_anchors}."
            )

        distance_anchors = anchors.float()
        if knn_points is not None:
            squared_distances = knn_points(
                distance_anchors,
                distance_anchors,
                K=self.knn + 1,
                return_nn=False,
            ).dists[..., 1:]
            return squared_distances.clamp_min(0).sqrt().mean(
                dim=-1,
                keepdim=True,
            ).to(dtype=anchors.dtype)

        spacing_chunks = []
        for start in range(0, num_anchors, self.knn_chunk_size):
            end = min(start + self.knn_chunk_size, num_anchors)
            distances = torch.cdist(
                distance_anchors[:, start:end],
                distance_anchors,
            )
            global_index = torch.arange(start, end, device=anchors.device)
            self_indices = global_index[None, :, None].expand(
                anchors.shape[0],
                -1,
                1,
            )
            distances = distances.scatter(2, self_indices, torch.inf)
            nearest = distances.topk(self.knn, dim=-1, largest=False).values
            spacing_chunks.append(nearest.mean(dim=-1, keepdim=True))

        return torch.cat(spacing_chunks, dim=1).to(dtype=anchors.dtype)

    def forward(self, anchors: torch.Tensor) -> AnchorGeometryEncoding:
        if anchors.ndim != 3 or anchors.shape[-1] != 3:
            raise ValueError(
                f"Expected anchors with shape [B, N, 3], got {tuple(anchors.shape)}."
            )

        spacing = self._local_spacing(anchors).clamp_min(self.spacing_eps)
        geometry_input = torch.cat(
            (self._fourier_encode(anchors), spacing.log()),
            dim=-1,
        )
        return AnchorGeometryEncoding(
            query=self.query_mlp(geometry_input),
            spacing=spacing,
        )
