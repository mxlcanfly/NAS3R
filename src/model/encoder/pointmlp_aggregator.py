import torch
import torch.nn as nn
from einops import rearrange
from pytorch3d.ops import knn_gather, knn_points


class ResidualPointMLPBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
            nn.LayerNorm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PointMLPAggregator(nn.Module):
    """PointMLP-style local aggregation that preserves the input point count."""

    def __init__(
        self,
        channels: int,
        k_neighbors: int = 16,
        anchor_stride: int = 8,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.anchor_stride = anchor_stride
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channels + 3))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channels + 3))
        self.transfer = nn.Sequential(
            nn.Linear(2 * channels + 3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualPointMLPBlock(channels) for _ in range(num_blocks)]
        )

    def _anchor_indices(
        self,
        v: int,
        h: int,
        w: int,
        surfaces: int,
        samples_per_pixel: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_indices = torch.arange(
            v * h * w * surfaces * samples_per_pixel,
            device=device,
        ).reshape(v, h, w, surfaces, samples_per_pixel)
        anchor_indices = point_indices[
            :, ::self.anchor_stride, ::self.anchor_stride
        ].reshape(-1)

        anchor_h = (h + self.anchor_stride - 1) // self.anchor_stride
        anchor_w = (w + self.anchor_stride - 1) // self.anchor_stride
        anchor_grid = torch.arange(
            v * anchor_h * anchor_w * surfaces * samples_per_pixel,
            device=device,
        ).reshape(v, anchor_h, anchor_w, surfaces, samples_per_pixel)
        rows = torch.arange(h, device=device) // self.anchor_stride
        cols = torch.arange(w, device=device) // self.anchor_stride
        point_to_anchor = anchor_grid[:, rows[:, None], cols[None, :]].reshape(-1)
        return anchor_indices, point_to_anchor

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        b: int,
        v: int,
        h: int,
        w: int,
        surfaces: int,
        samples_per_pixel: int,
    ) -> torch.Tensor:
        features = rearrange(
            features,
            "(b v hw s spp) c -> b (v hw s spp) c",
            b=b,
            v=v,
            hw=h * w,
            s=surfaces,
            spp=samples_per_pixel,
        )
        coordinates = rearrange(
            coordinates,
            "b v hw s spp xyz -> b (v hw s spp) xyz",
        ).detach()
        anchor_indices, point_to_anchor = self._anchor_indices(
            v,
            h,
            w,
            surfaces,
            samples_per_pixel,
            features.device,
        )
        anchor_coordinates = coordinates[:, anchor_indices]
        anchor_features = features[:, anchor_indices]

        k = min(self.k_neighbors, coordinates.shape[1])
        with torch.no_grad():
            neighbor_indices = knn_points(
                anchor_coordinates.float(),
                coordinates.float(),
                K=k,
                return_nn=False,
                return_sorted=False,
            ).idx

        neighbor_coordinates = knn_gather(coordinates, neighbor_indices)
        neighbor_features = knn_gather(features, neighbor_indices)
        grouped_features = torch.cat(
            [neighbor_features, neighbor_coordinates],
            dim=-1,
        )
        anchor_features_with_coordinates = torch.cat(
            [anchor_features, anchor_coordinates],
            dim=-1,
        ).unsqueeze(2)
        centered_features = grouped_features - anchor_features_with_coordinates
        feature_std = centered_features.flatten(1).std(dim=1).view(b, 1, 1, 1)
        normalized_features = centered_features / feature_std.clamp_min(1e-5)
        normalized_features = (
            normalized_features * self.affine_alpha + self.affine_beta
        )

        local_features = torch.cat(
            [
                normalized_features,
                anchor_features.unsqueeze(2).expand(-1, -1, k, -1),
            ],
            dim=-1,
        )
        local_features = self.blocks(self.transfer(local_features))
        anchor_features = local_features.max(dim=2).values

        aggregated_features = anchor_features[:, point_to_anchor]
        aggregated_features = rearrange(aggregated_features, "b n c -> (b n) c")
        original_features = rearrange(features, "b n c -> (b n) c")
        return original_features + aggregated_features
