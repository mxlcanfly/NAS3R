import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.body = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.body(x)


class AnchorAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        expansion_ratio: int = 4,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nn.RMSNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.RMSNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * expansion_ratio),
            nn.GELU(),
            nn.Linear(channels * expansion_ratio, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = rearrange(
            q,
            "b n (h d) -> b h n d",
            h=self.num_heads,
            d=self.head_dim,
        )
        k = rearrange(
            k,
            "b n (h d) -> b h n d",
            h=self.num_heads,
            d=self.head_dim,
        )
        v = rearrange(
            v,
            "b n (h d) -> b h n d",
            h=self.num_heads,
            d=self.head_dim,
        )
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = rearrange(
            attended,
            "b h n d -> b n (h d)",
            h=self.num_heads,
            n=length,
            d=self.head_dim,
        )
        x = x + self.proj(attended)
        return x + self.mlp(self.norm2(x))


class AnchorFeatureUNet(nn.Module):
    """HiSplat-style U-Net for geometry-visual anchor embeddings."""

    def __init__(
        self,
        feature_dim: int = 256,
        base_dim: int = 128,
        num_attention_blocks: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(feature_dim, base_dim, 3, padding=1)
        self.encoder0 = ResidualBlock2d(base_dim, base_dim)
        self.down0 = nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1)
        self.encoder1 = ResidualBlock2d(base_dim, base_dim)
        self.down1 = nn.Conv2d(
            base_dim,
            base_dim,
            3,
            stride=2,
            padding=1,
        )
        self.bottleneck = ResidualBlock2d(base_dim, base_dim)
        self.attention = nn.ModuleList(
            [
                AnchorAttentionBlock(
                    channels=base_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_attention_blocks)
            ]
        )
        self.up1 = nn.ConvTranspose2d(
            base_dim,
            base_dim,
            4,
            stride=2,
            padding=1,
        )
        self.decoder1 = ResidualBlock2d(base_dim * 2, base_dim)
        self.up0 = nn.ConvTranspose2d(
            base_dim,
            base_dim,
            4,
            stride=2,
            padding=1,
        )
        self.decoder0 = ResidualBlock2d(base_dim * 2, base_dim)
        self.output_proj = nn.Conv2d(base_dim, feature_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        features: Tensor,
        num_views: int,
        height: int,
        width: int,
        num_surfaces: int,
        samples_per_pixel: int,
    ) -> Tensor:
        grid = rearrange(
            features,
            "b (v h w srf spp) c -> (b srf spp v) c h w",
            v=num_views,
            h=height,
            w=width,
            srf=num_surfaces,
            spp=samples_per_pixel,
        )
        skip0 = self.encoder0(self.input_proj(grid))
        skip1 = self.encoder1(self.down0(skip0))
        hidden = self.bottleneck(self.down1(skip1))

        bottleneck_height, bottleneck_width = hidden.shape[-2:]
        tokens = rearrange(
            hidden,
            "(b srf spp v) c h w -> (b srf spp) (v h w) c",
            b=features.shape[0],
            srf=num_surfaces,
            spp=samples_per_pixel,
            v=num_views,
        )
        for block in self.attention:
            tokens = block(tokens)
        hidden = rearrange(
            tokens,
            "(b srf spp) (v h w) c -> (b srf spp v) c h w",
            b=features.shape[0],
            srf=num_surfaces,
            spp=samples_per_pixel,
            v=num_views,
            h=bottleneck_height,
            w=bottleneck_width,
        )

        hidden = self.up1(hidden)
        if hidden.shape[-2:] != skip1.shape[-2:]:
            hidden = F.interpolate(
                hidden,
                size=skip1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        hidden = self.decoder1(torch.cat([hidden, skip1], dim=1))
        hidden = self.up0(hidden)
        if hidden.shape[-2:] != skip0.shape[-2:]:
            hidden = F.interpolate(
                hidden,
                size=skip0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        hidden = self.decoder0(torch.cat([hidden, skip0], dim=1))
        residual = self.output_proj(hidden)
        residual = rearrange(
            residual,
            "(b srf spp v) c h w -> b (v h w srf spp) c",
            b=features.shape[0],
            srf=num_surfaces,
            spp=samples_per_pixel,
            v=num_views,
        )
        return features + residual
