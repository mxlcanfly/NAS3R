from dataclasses import dataclass
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from ...geometry.projection import homogenize_points, transform_world2cam


_PTV3_ROOT = Path("/space0/mengxl")
if str(_PTV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_PTV3_ROOT))

from PointTransformerV3.model import Block, Point, PointSequential  # noqa: E402


@dataclass
class GDStyleAnchorDensifierCfg:
    enabled: bool = True
    lr_gs_feature_dim: int = 256
    sampled_feature_dim: int = 176
    hidden_dim: int = 160
    cross_attn_heads: int = 16
    point_depth: int = 2
    point_heads: int = 8
    point_patch_size: int = 48
    point_mlp_ratio: float = 4.0
    num_offsets: int = 8
    offset_init_radius_scale: float = 0.5
    # Offsets are norm-bounded by a sphere inscribed in the anchor's LR pixel
    # frustum: radius = scale * z / (f_px * lr_grid). 0.5 reaches the cell border.
    offset_pixel_radius_scale: float = 0.5
    offset_init_pattern: str = "sphere"
    offset_coordinate_frame: str = "world"


class PointFeatureAggregator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders: bool = True,
    ) -> None:
        super().__init__()
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
                    cpe_indice_key=f"gd_anchor_point_stage{i}",
                    enable_flash=False,
                    upcast_attention=False,
                    upcast_softmax=False,
                ),
                name=f"block{i}",
            )

    @staticmethod
    def _batch_offsets(batch_size: int, points_per_batch: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            1,
            batch_size + 1,
            device=device,
            dtype=torch.long,
        ) * points_per_batch

    @staticmethod
    def _make_grid_coord(anchors: torch.Tensor) -> torch.Tensor:
        coords = rearrange(anchors.detach(), "b n c -> (b n) c")
        coord_range = coords.max(dim=0).values - coords.min(dim=0).values
        quant_size = coord_range[torch.isfinite(coord_range) & (coord_range > 0)]
        if quant_size.numel() == 0:
            quant_size = anchors.new_tensor(1.0)
        else:
            quant_size = (quant_size.median() / 256.0).clamp_min(1e-6)
        return torch.div(
            coords - coords.min(dim=0).values,
            quant_size,
            rounding_mode="trunc",
        ).int()

    def forward(self, anchors: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if anchors.shape[:2] != features.shape[:2]:
            raise ValueError(
                "anchors and features must share batch/point axes, got "
                f"{tuple(anchors.shape)} and {tuple(features.shape)}."
            )
        b, n, _ = anchors.shape
        point = Point(
            {
                "coord": rearrange(anchors, "b n c -> (b n) c"),
                "grid_coord": self._make_grid_coord(anchors),
                "feat": self.input_proj(rearrange(features, "b n c -> (b n) c")),
                "offset": self._batch_offsets(b, n, anchors.device),
            }
        )
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.blocks(point)
        return rearrange(point.feat, "(b n) c -> b n c", b=b, n=n)


class GDStyleAnchorDensifier(nn.Module):
    def __init__(self, cfg: GDStyleAnchorDensifierCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(cfg.lr_gs_feature_dim),
            nn.Linear(cfg.lr_gs_feature_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.query_norm = nn.LayerNorm(cfg.hidden_dim)
        self.kv_mlp = nn.Sequential(
            nn.LayerNorm(cfg.sampled_feature_dim),
            nn.Linear(cfg.sampled_feature_dim, cfg.hidden_dim),
            nn.ReLU(),
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
        self.post_attn = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.point_aggregator = PointFeatureAggregator(
            in_channels=cfg.hidden_dim * 2,
            hidden_channels=cfg.hidden_dim,
            depth=cfg.point_depth,
            num_heads=cfg.point_heads,
            patch_size=cfg.point_patch_size,
            mlp_ratio=cfg.point_mlp_ratio,
        )
        self.in_norm = nn.LayerNorm(cfg.hidden_dim)
        self.offset_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_offsets * 3),
        )
        self._init_linear(self.query_mlp)
        self._init_linear(self.kv_mlp)
        self._init_linear(self.post_attn)
        self._init_offset_mlp()

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _fibonacci_sphere(num_points: int) -> torch.Tensor:
        if num_points <= 0:
            raise ValueError(f"num_offsets must be positive, got {num_points}.")
        if num_points == 1:
            return torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        index = torch.arange(num_points, dtype=torch.float32)
        z = 1.0 - 2.0 * index / (num_points - 1)
        radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        theta = index * math.pi * (3.0 - math.sqrt(5.0))
        directions = torch.stack(
            [radius * theta.cos(), radius * theta.sin(), z],
            dim=-1,
        )
        return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _circle_directions(num_points: int) -> torch.Tensor:
        if num_points <= 0:
            raise ValueError(f"num_offsets must be positive, got {num_points}.")
        theta = torch.arange(num_points, dtype=torch.float32) * (2.0 * math.pi / num_points)
        directions = torch.stack(
            [theta.cos(), theta.sin(), torch.zeros_like(theta)],
            dim=-1,
        )
        return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _init_offset_mlp(self) -> None:
        first_layer, last_layer = self.offset_mlp[0], self.offset_mlp[-1]
        nn.init.xavier_uniform_(first_layer.weight)
        nn.init.zeros_(first_layer.bias)
        nn.init.zeros_(last_layer.weight)

        init_scale = min(max(self.cfg.offset_init_radius_scale, 1e-4), 0.999)
        if self.cfg.offset_init_pattern == "sphere":
            directions = self._fibonacci_sphere(self.cfg.num_offsets)
        elif self.cfg.offset_init_pattern == "circle":
            directions = self._circle_directions(self.cfg.num_offsets)
        else:
            raise ValueError(
                "offset_init_pattern must be 'sphere' or 'circle', got "
                f"{self.cfg.offset_init_pattern!r}."
            )
        # Offsets are parameterized as tanh(|raw|) * radius * raw/|raw|, so a raw
        # vector of atanh(s) * unit_direction yields an initial offset of
        # s * radius along that direction (children start evenly spread at a
        # fraction s of the pixel-sphere radius).
        bias = math.atanh(init_scale) * directions
        with torch.no_grad():
            last_layer.bias.copy_(bias.reshape(-1))

    def _pixel_sphere_radius(
        self,
        anchors: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        lr_grid_size: tuple[int, int],
    ) -> torch.Tensor:
        """Radius of the sphere inscribed in each anchor's own-view LR pixel frustum.

        A sphere of radius <= 0.5 * pixel_world_size centered on the anchor projects
        inside the anchor's LR pixel cell from any viewpoint, so norm-bounding the
        offsets by this radius keeps children within their parent pixel.
        """
        lr_height, lr_width = lr_grid_size
        with torch.no_grad():
            cam_points = transform_world2cam(
                homogenize_points(anchors.detach()),
                extrinsics[:, :, None],
            )
            x, y = cam_points[..., 0], cam_points[..., 1]
            anchor_z = cam_points[..., 2].clamp_min(1e-3)
            fx = intrinsics[..., 0, 0].clamp_min(1e-6)[:, :, None]  # normalized by image width
            fy = intrinsics[..., 1, 1].clamp_min(1e-6)[:, :, None]  # normalized by image height
            # Exact distance from the anchor to its pixel cell's boundary planes.
            # Off-axis cells are oblique cones: the plane through the camera center
            # for image column u_b has normal (fx, 0, -(u_b - 0.5)); using the outer
            # boundary (anchor u plus half a cell) makes the bound exact.
            u_off = (fx * x / anchor_z).abs() + 0.5 / lr_width
            v_off = (fy * y / anchor_z).abs() + 0.5 / lr_height
            radius_x = anchor_z / (lr_width * torch.sqrt(fx.square() + u_off.square()))
            radius_y = anchor_z / (lr_height * torch.sqrt(fy.square() + v_off.square()))
            radius = self.cfg.offset_pixel_radius_scale * torch.minimum(radius_x, radius_y)
            return radius[..., None]

    def forward(
        self,
        anchors: torch.Tensor,
        lr_gs_features: torch.Tensor,
        sampled_features: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        lr_grid_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if anchors.shape[:3] != lr_gs_features.shape[:3]:
            raise ValueError(
                "anchors and lr_gs_features must share [B, V, N], got "
                f"{tuple(anchors.shape)} and {tuple(lr_gs_features.shape)}."
            )
        if sampled_features.shape[:3] != anchors.shape[:3]:
            raise ValueError(
                "sampled_features must share [B, V, N] with anchors, got "
                f"{tuple(sampled_features.shape)} and {tuple(anchors.shape)}."
            )
        if extrinsics.shape[:2] != anchors.shape[:2] or extrinsics.shape[-2:] != (4, 4):
            raise ValueError(
                "extrinsics must have shape [B, V, 4, 4] matching anchors, got "
                f"{tuple(extrinsics.shape)} and {tuple(anchors.shape)}."
            )
        if self.cfg.offset_coordinate_frame not in ("camera", "world"):
            raise ValueError(
                "offset_coordinate_frame must be 'camera' or 'world', got "
                f"{self.cfg.offset_coordinate_frame!r}."
            )

        b, v, n, _ = anchors.shape
        query = self.query_mlp(lr_gs_features)
        query = self.query_norm(query)
        kv = self.kv_mlp(sampled_features)

        query_flat = rearrange(query, "b v n c -> (b v n) 1 c")
        kv_flat = rearrange(kv, "b v n t c -> (b v n) t c")
        fine, _ = self.cross_attn(query_flat, kv_flat, kv_flat, need_weights=False)
        fine = self.post_attn(fine.squeeze(1))
        fine = rearrange(fine, "(b v n) c -> b v n c", b=b, v=v, n=n)

        point_features = torch.cat([query, fine], dim=-1)
        point_features = self.point_aggregator(
            rearrange(anchors, "b v n xyz -> b (v n) xyz"),
            rearrange(point_features, "b v n c -> b (v n) c"),
        )
        point_features = rearrange(point_features, "b (v n) c -> b v n c", v=v, n=n)
        point_features = self.in_norm(point_features)

        raw_offsets = self.offset_mlp(point_features)
        raw_offsets = rearrange(raw_offsets, "b v n (k xyz) -> b v n k xyz", k=self.cfg.num_offsets, xyz=3)
        offset_radius = self._pixel_sphere_radius(anchors, extrinsics, intrinsics, lr_grid_size)
        # Norm-bounded offsets: |offset| <= radius regardless of direction, so the
        # bound must be on the vector norm (a per-component tanh would give a box
        # whose diagonal exceeds the pixel sphere).
        raw_norm = raw_offsets.norm(dim=-1, keepdim=True)
        offset_directions = raw_offsets / raw_norm.clamp_min(1e-6)
        local_offsets = raw_norm.tanh() * offset_radius[:, :, :, None, :] * offset_directions
        if self.cfg.offset_coordinate_frame == "camera":
            rotation_c2w = extrinsics[..., :3, :3]
            offsets = torch.einsum("bvij,bvnkj->bvnki", rotation_c2w, local_offsets)
        else:
            offsets = local_offsets
        child_centers = anchors[:, :, :, None] + offsets

        return {
            "point_features": point_features,
            "local_offsets": local_offsets,
            "offsets": offsets,
            "child_centers": child_centers,
            "offset_radius": offset_radius,
        }
