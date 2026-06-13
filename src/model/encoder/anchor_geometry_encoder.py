import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AnchorGeometryEncoding:
    query: torch.Tensor
    spacing: torch.Tensor


class AnchorGeometryEncoder(nn.Module):
    """Encode anchor position and its mean KNN spacing."""

    def __init__(
        self,
        num_frequencies: int = 6,
        query_dim: int = 128,
        hidden_dim: int = 256,
        num_neighbors: int = 8,
        knn_chunk_size: int = 1024,
        spacing_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive.")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive.")
        if knn_chunk_size <= 0:
            raise ValueError("knn_chunk_size must be positive.")
        if spacing_eps <= 0:
            raise ValueError("spacing_eps must be positive.")

        self.num_frequencies = num_frequencies
        self.num_neighbors = num_neighbors
        self.knn_chunk_size = knn_chunk_size
        self.spacing_eps = spacing_eps

        position_dim = 3 * (1 + 2 * num_frequencies)
        spacing_dim = 1 + 2 * num_frequencies
        self.query_mlp = nn.Sequential(
            nn.Linear(position_dim + spacing_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
            nn.LayerNorm(query_dim),
        )

    def _fourier_encode(self, values: torch.Tensor) -> torch.Tensor:
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies,
            device=values.device,
            dtype=values.dtype,
        )
        angles = values[..., None] * frequencies * math.pi
        periodic = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.cat((values, periodic.flatten(start_dim=-2)), dim=-1)

    @torch.no_grad()
    def _compute_mean_knn_spacing(self, anchors: torch.Tensor) -> torch.Tensor:
        b, num_anchors, _ = anchors.shape
        if num_anchors <= 1:
            return torch.full(
                (b, num_anchors, 1),
                self.spacing_eps,
                device=anchors.device,
                dtype=anchors.dtype,
            )

        num_neighbors = min(self.num_neighbors, num_anchors - 1)
        anchors_float = anchors.detach().float()
        spacing_per_batch = []
        for batch_anchors in anchors_float:
            spacing_chunks = []
            for start in range(0, num_anchors, self.knn_chunk_size):
                end = min(start + self.knn_chunk_size, num_anchors)
                distances = torch.cdist(
                    batch_anchors[start:end],
                    batch_anchors,
                )
                local_rows = torch.arange(
                    end - start,
                    device=anchors.device,
                )
                global_rows = torch.arange(
                    start,
                    end,
                    device=anchors.device,
                )
                distances[local_rows, global_rows] = torch.inf
                nearest_distances = distances.topk(
                    k=num_neighbors,
                    dim=-1,
                    largest=False,
                ).values
                spacing_chunks.append(nearest_distances.mean(dim=-1))
            spacing_per_batch.append(torch.cat(spacing_chunks, dim=0))

        spacing = torch.stack(spacing_per_batch, dim=0)
        spacing = spacing.clamp_min(self.spacing_eps).unsqueeze(-1)
        return spacing.to(dtype=anchors.dtype)

    def forward(self, anchors: torch.Tensor) -> AnchorGeometryEncoding:
        if anchors.ndim != 3 or anchors.shape[-1] != 3:
            raise ValueError(
                f"Expected anchors with shape [B, N, 3], got {tuple(anchors.shape)}."
            )

        spacing = self._compute_mean_knn_spacing(anchors)
        geometry_input = torch.cat(
            (
                self._fourier_encode(anchors),
                self._fourier_encode(spacing.log()),
            ),
            dim=-1,
        )
        return AnchorGeometryEncoding(
            query=self.query_mlp(geometry_input),
            spacing=spacing,
        )


class PointGeometryEncoder(nn.Module):
    """Encode world-space points without an explicit neighborhood-density cue."""

    def __init__(
        self,
        num_frequencies: int = 6,
        output_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        position_dim = 3 * (1 + 2 * num_frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _fourier_encode(self, points: torch.Tensor) -> torch.Tensor:
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies,
            device=points.device,
            dtype=points.dtype,
        )
        angles = points[..., None] * frequencies * math.pi
        periodic = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.cat((points, periodic.flatten(start_dim=-2)), dim=-1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"Expected points with shape [B, N, 3], got {tuple(points.shape)}."
            )
        return self.mlp(self._fourier_encode(points))
