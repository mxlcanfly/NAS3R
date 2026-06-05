from pathlib import Path
import sys

import torch
from einops import rearrange
from torch import nn


class PureTorchKNNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        knn: int = 16,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.knn = knn
        self.chunk_size = chunk_size
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def _knn_indices(self, coord: torch.Tensor) -> torch.Tensor:
        num_points = coord.shape[0]
        k = min(self.knn, num_points)
        indices = []
        coord_float = coord.float()
        for start in range(0, num_points, self.chunk_size):
            end = min(start + self.chunk_size, num_points)
            dist = torch.cdist(coord_float[start:end], coord_float)
            indices.append(dist.topk(k=k, dim=-1, largest=False).indices)
        return torch.cat(indices, dim=0)

    def forward(self, coord: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        residual = feature
        feature_norm = self.norm1(feature)
        q, k, v = self.qkv(feature_norm).chunk(3, dim=-1)
        knn_idx = self._knn_indices(coord)
        k_neighbors = k[knn_idx]
        v_neighbors = v[knn_idx]
        scale = q.shape[-1] ** -0.5
        attn = (q[:, None] * k_neighbors).sum(dim=-1) * scale
        attn = attn.softmax(dim=-1)
        feature = residual + self.proj((attn[..., None] * v_neighbors).sum(dim=1))
        feature = feature + self.mlp(self.norm2(feature))
        return feature


class PureTorchAnchorPointTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int = 2,
        knn: int = 16,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PureTorchKNNBlock(
                    channels=channels,
                    knn=knn,
                    chunk_size=chunk_size,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, anchors: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        for coord, feature in zip(anchors, features, strict=True):
            for block in self.blocks:
                feature = block(coord, feature)
            outputs.append(feature)
        return torch.stack(outputs, dim=0)


class AnchorLitePTFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        geometry_dim: int = 128,
        token_dim: int = 256,
        litept_path: str = "/space0/mengxl/LitePT-main",
        use_full_litept: bool = True,
        litept_grid_size: float = 0.02,
        litept_output_dim: int = 72,
        litept_enc_channels: tuple[int, ...] = (36, 72, 144, 252, 504),
        litept_enc_num_head: tuple[int, ...] = (2, 4, 8, 14, 28),
        litept_dec_channels: tuple[int, ...] = (72, 72, 144, 252),
        litept_dec_num_head: tuple[int, ...] = (4, 4, 8, 14),
        fallback_blocks: int = 2,
        fallback_knn: int = 16,
        fallback_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.use_full_litept = use_full_litept
        self.litept_grid_size = litept_grid_size
        self.last_litept_error: str | None = None

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim + geometry_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.litept = None
        if use_full_litept:
            self.litept = self._load_litept(
                litept_path,
                token_dim,
                litept_enc_channels=tuple(litept_enc_channels),
                litept_enc_num_head=tuple(litept_enc_num_head),
                litept_dec_channels=tuple(litept_dec_channels),
                litept_dec_num_head=tuple(litept_dec_num_head),
            )

        self.fallback_pt = PureTorchAnchorPointTransformer(
            channels=token_dim,
            num_blocks=fallback_blocks,
            knn=fallback_knn,
            chunk_size=fallback_chunk_size,
        )
        self.litept_out_proj = nn.Sequential(
            nn.Linear(litept_output_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def _load_litept(
        self,
        litept_path: str,
        token_dim: int,
        litept_enc_channels: tuple[int, ...],
        litept_enc_num_head: tuple[int, ...],
        litept_dec_channels: tuple[int, ...],
        litept_dec_num_head: tuple[int, ...],
    ) -> nn.Module | None:
        root = Path(litept_path)
        if not (root / "litept" / "model.py").is_file():
            root = root / "LitePT-main"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from litept.model import LitePT
        except Exception:
            return None
        return LitePT(
            in_channels=token_dim,
            enc_channels=litept_enc_channels,
            enc_num_head=litept_enc_num_head,
            dec_channels=litept_dec_channels,
            dec_num_head=litept_dec_num_head,
        )

    def _run_full_litept(
        self,
        anchors: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.litept is None:
            return None
        if not anchors.is_cuda:
            return None
        b, n, _ = anchors.shape
        coord = rearrange(anchors, "b n c -> (b n) c").float()
        feat = rearrange(tokens, "b n c -> (b n) c").float()
        offset = torch.arange(1, b + 1, device=anchors.device, dtype=torch.long) * n
        try:
            point = self.litept(
                {
                    "coord": coord,
                    "grid_size": torch.tensor(
                        self.litept_grid_size,
                        device=anchors.device,
                        dtype=coord.dtype,
                    ),
                    "feat": feat,
                    "offset": offset,
                }
            )
        except Exception as exc:
            self.last_litept_error = f"{type(exc).__name__}: {exc}"
            return None
        litept_feat = point.feat
        if litept_feat.shape[0] != feat.shape[0]:
            inverse = getattr(point, "inverse", None)
            if inverse is None:
                inverse = getattr(point, "unpooling_inverse", None)
            if inverse is None or inverse.shape[0] != feat.shape[0]:
                self.last_litept_error = (
                    "LitePT changed point count but did not return a valid inverse "
                    f"mapping: feat={tuple(litept_feat.shape)}, inverse="
                    f"{None if inverse is None else tuple(inverse.shape)}, "
                    f"input={tuple(feat.shape)}."
                )
                return None
            litept_feat = litept_feat[inverse]
        self.last_litept_error = None
        return rearrange(
            self.litept_out_proj(litept_feat.float()),
            "(b n) c -> b n c",
            b=b,
            n=n,
        )

    def forward(
        self,
        anchors: torch.Tensor,
        anchor_features: torch.Tensor,
        geometry_query: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.input_proj(torch.cat([anchor_features, geometry_query], dim=-1))
        litept_delta = self._run_full_litept(anchors, tokens)
        if litept_delta is None:
            if self.use_full_litept:
                raise RuntimeError(
                    "Full LitePT failed, so the memory-heavy PyTorch KNN fallback "
                    "was not executed. Original LitePT error: "
                    f"{self.last_litept_error or 'LitePT is unavailable or input is not CUDA.'}"
                )
            litept_delta = self.fallback_pt(anchors, tokens)
        # tokens = self.output_norm(tokens + litept_delta)
        tokens = tokens + litept_delta
        return tokens
