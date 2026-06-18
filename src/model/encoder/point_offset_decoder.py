import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange


_PTV3_ROOT = Path("/space0/mengxl")
if str(_PTV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_PTV3_ROOT))

from PointTransformerV3.model import Block, Point, PointSequential  # noqa: E402


class PointOffsetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        k_offsets: int = 8,
        grid_size: float = 0.02,
        offset_scale: float = 0.1,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.offset_scale = offset_scale
        self.k_offsets = k_offsets
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.blocks = PointSequential()
        for i in range(depth):
            self.blocks.add(
                Block(
                    channels=hidden_channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    pre_norm=True,
                    order_index=i % len(self.order),
                    cpe_indice_key=f"point_offset_stage{i}",
                    enable_flash=False,
                    upcast_attention=False,
                    upcast_softmax=False,
                ),
                name=f"block{i}",
            )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 3 * k_offsets),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    @staticmethod
    def _batch_offsets(batch_size: int, points_per_batch: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            1,
            batch_size + 1,
            device=device,
            dtype=torch.long,
        ) * points_per_batch

    def forward(
        self,
        anchors: torch.Tensor,
        features: torch.Tensor,
        offset_radius: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if anchors.shape[:2] != features.shape[:2]:
            raise ValueError(
                "anchors and features must share batch/point axes, got "
                f"{tuple(anchors.shape)} and {tuple(features.shape)}."
            )
        b, n, _ = anchors.shape
        point = Point(
            {
                "coord": rearrange(anchors, "b n c -> (b n) c"),
                "feat": self.input_proj(rearrange(features, "b n c -> (b n) c")),
                "offset": self._batch_offsets(b, n, anchors.device),
                "grid_size": self.grid_size,
            }
        )
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.blocks(point)

        offsets = self.offset_head(point.feat)
        offsets = rearrange(offsets, "(b n) (k xyz) -> b n k xyz", b=b, n=n, k=self.k_offsets, xyz=3)
        if offset_radius is None:
            offset_radius = torch.full(
                (b, n, 1),
                self.grid_size,
                dtype=anchors.dtype,
                device=anchors.device,
            )
        elif offset_radius.shape[:2] != anchors.shape[:2]:
            raise ValueError(
                "offset_radius must share batch/point axes with anchors, got "
                f"{tuple(offset_radius.shape)} and {tuple(anchors.shape)}."
            )
        if offset_radius.shape[-1] not in (1, 3):
            raise ValueError(
                "offset_radius must have last dimension 1 or 3, got "
                f"{tuple(offset_radius.shape)}."
            )
        offset_radius = offset_radius[:, :, None].clamp_min(1e-6)
        offsets = self.offset_scale * offset_radius * torch.tanh(offsets)
        child_centers = anchors[:, :, None] + offsets
        return {
            "offsets": offsets,
            "child_centers": child_centers,
            "parent_centers": anchors,
            "parent_features": rearrange(point.feat, "(b n) c -> b n c", b=b, n=n),
            "offset_radius": offset_radius.squeeze(2),
        }
