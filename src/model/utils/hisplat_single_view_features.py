from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class HiSplatSingleViewFeatureCfg:
    enabled: bool = True
    resunet_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    croco_dim: int = 1024
    out_dim_64: int = 128
    out_dim_256: int = 64


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                norm_layer(planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class ResUnetEncoder(nn.Module):
    def __init__(
        self,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        feature_dims: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.layer0 = self._make_layer(feature_dims[0], feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer1 = self._make_layer(feature_dims[0], feature_dims[1], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], feature_dims[2], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(feature_dims[2], feature_dims[2], stride=1, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], feature_dims[2], 1, 1, 0)
        self._init_weights()

    @staticmethod
    def _make_layer(
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, norm_layer=norm_layer, stride=stride, dilation=dilation),
            ResidualBlock(out_dim, out_dim, norm_layer=norm_layer, stride=1, dilation=dilation),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer0(x)
        features.append(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        x = self.conv2(x)
        features.append(x)
        return features


class ResUnetDecoder(nn.Module):
    def __init__(
        self,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        feature_dims: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.decode_layer1 = self._make_fine_layer(2 * feature_dims[-1], feature_dims[-1], feature_dims[-2], norm_layer=norm_layer)
        self.decode_layer0 = self._make_fine_layer(2 * feature_dims[-2], feature_dims[-2], feature_dims[-3], norm_layer=norm_layer)
        self.out_layer0 = self._make_fine_layer(2 * feature_dims[0], feature_dims[0], feature_dims[0], norm_layer=norm_layer)
        self.out_layer1 = self._make_fine_layer(2 * feature_dims[1], feature_dims[1], feature_dims[1], norm_layer=norm_layer)
        self.out_layer2 = self._make_fine_layer(2 * feature_dims[2], feature_dims[2], feature_dims[2], norm_layer=norm_layer)
        self._init_weights()

    @staticmethod
    def _make_fine_layer(
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_dim, mid_dim, norm_layer=norm_layer, stride=stride, dilation=dilation),
            ResidualBlock(mid_dim, out_dim, norm_layer=norm_layer, stride=1, dilation=dilation),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, feature_list: list[Tensor], croco_feature_list: list[Tensor] | None = None) -> list[Tensor]:
        out_features = []
        x = feature_list[-1]
        if croco_feature_list is not None:
            x = x + croco_feature_list[0]
        x = torch.cat([x, feature_list[-2]], dim=1)
        out_features.append(self.out_layer2(x))

        x = self.decode_layer1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if croco_feature_list is not None:
            x = x + croco_feature_list[1]
        x = torch.cat([x, feature_list[-3]], dim=1)
        out_features.append(self.out_layer1(x))

        x = self.decode_layer0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if croco_feature_list is not None:
            x = x + croco_feature_list[2]
        x = torch.cat([x, feature_list[-4]], dim=1)
        out_features.append(self.out_layer0(x))
        return out_features


class ResUnet(nn.Module):
    def __init__(
        self,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        feature_dims: tuple[int, int, int] = (32, 64, 128),
        croco_feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResUnetEncoder(norm_layer=norm_layer, feature_dims=feature_dims)
        self.decoder = ResUnetDecoder(norm_layer=norm_layer, feature_dims=feature_dims)
        self.up_croco_cnn = nn.ModuleList(
            [
                nn.Conv2d(croco_feature_dim, feature_dims[-1], 1, bias=False),
                nn.Conv2d(croco_feature_dim, feature_dims[-2], 1, bias=False),
                nn.Conv2d(croco_feature_dim, feature_dims[-3], 1, bias=False),
            ]
        )

    def forward(self, x: Tensor, croco_feature: Tensor | None = None) -> list[Tensor]:
        feature_list = self.encoder(x)
        croco_feature_list = None
        if croco_feature is not None:
            croco_feature_list = []
            for i, conv in enumerate(self.up_croco_cnn):
                target = feature_list[len(feature_list) - i - 2]
                croco_feature_list.append(
                    conv(
                        F.interpolate(
                            croco_feature,
                            size=target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                )
        return self.decoder(feature_list, croco_feature_list)


class SingleViewSRFeatureExtractor(nn.Module):
    """Single-view SR feature extractor using CroCo encoder tokens and HiSplat ResUNet."""

    def __init__(self, cfg: HiSplatSingleViewFeatureCfg) -> None:
        super().__init__()
        self.cfg = cfg
        resunet_dims = tuple(cfg.resunet_dims)
        self.croco_proj = nn.Conv2d(cfg.croco_dim, 64, 1)
        self.resunet = ResUnet(feature_dims=resunet_dims, croco_feature_dim=64)
        self.fuse64 = nn.Sequential(
            nn.Conv2d(resunet_dims[-1] + 64, cfg.out_dim_64, 1),
            nn.GELU(),
            nn.Conv2d(cfg.out_dim_64, cfg.out_dim_64, 3, padding=1),
        )
        self.fuse256 = nn.Sequential(
            nn.Conv2d(resunet_dims[0] + 64, cfg.out_dim_256, 1),
            nn.GELU(),
            nn.Conv2d(cfg.out_dim_256, cfg.out_dim_256, 3, padding=1),
        )

    @staticmethod
    def _croco_encoder_feature(backbone: Any, images: Tensor) -> Tensor:
        b, v, _, h, w = images.shape
        flat_images = rearrange(images, "b v c h w -> (b v) c h w")
        true_shape = torch.tensor(
            flat_images.shape[-2:],
            device=flat_images.device,
        )[None].repeat(b * v, 1)
        tokens, _, num_patch_tokens = backbone._encode_image(flat_images, true_shape)
        tokens = tokens[:, :num_patch_tokens]
        grid_h = h // backbone.patch_size
        grid_w = w // backbone.patch_size
        return rearrange(tokens, "(b v) (gh gw) c -> b v c gh gw", b=b, v=v, gh=grid_h, gw=grid_w)

    def forward(
        self,
        image_lr: Tensor,
        image_sr: Tensor,
        backbone: Any,
        croco_image_sr: Tensor | None = None,
    ) -> dict[str, Tensor]:
        croco_tokens = self._croco_encoder_feature(
            backbone,
            image_sr if croco_image_sr is None else croco_image_sr,
        )
        b, v, _, h_sr, w_sr = image_sr.shape
        croco_flat = rearrange(croco_tokens, "b v c h w -> (b v) c h w")
        croco_base = self.croco_proj(croco_flat)

        image_sr_flat = rearrange(image_sr, "b v c h w -> (b v) c h w")
        resunet_features = self.resunet(image_sr_flat, croco_base)
        resunet_64 = resunet_features[0]
        resunet_256 = resunet_features[-1]

        croco_64 = F.interpolate(croco_base, size=resunet_64.shape[-2:], mode="bilinear", align_corners=False)
        croco_256 = F.interpolate(croco_base, size=resunet_256.shape[-2:], mode="bilinear", align_corners=False)
        feature_64 = self.fuse64(torch.cat([resunet_64, croco_64], dim=1))
        feature_256 = self.fuse256(torch.cat([resunet_256, croco_256], dim=1))

        return {
            "image_lr": image_lr,
            "image_sr": image_sr,
            "feature_64": rearrange(feature_64, "(b v) c h w -> b v c h w", b=b, v=v),
            "feature_256": rearrange(feature_256, "(b v) c h w -> b v c h w", b=b, v=v),
            "croco_feature": croco_tokens,
            "resunet_feature_64": rearrange(resunet_64, "(b v) c h w -> b v c h w", b=b, v=v),
            "resunet_feature_256": rearrange(resunet_256, "(b v) c h w -> b v c h w", b=b, v=v),
        }
