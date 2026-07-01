import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .hisplat_multiview_transformer import MultiViewFeatureTransformer
from .hisplat_position import PositionEmbeddingSine


def feature_add_position_list(
    features_list: list[torch.Tensor],
    attn_splits: int,
    feature_channels: int,
) -> list[torch.Tensor]:
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        features_splits = [
            feature.reshape(-1, *feature.shape[-3:])
            for feature in features_list
        ]
        position = pos_enc(features_splits[0])
        return [
            feature + position
            for feature in features_splits
        ]

    position = pos_enc(features_list[0])
    return [
        feature + position
        for feature in features_list
    ]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        norm_layer=nn.InstanceNorm2d,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class ResUnetEncoder(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d, feature_dims=(32, 64, 128)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.layer0 = self._make_layer(feature_dims[0], feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer1 = self._make_layer(feature_dims[0], feature_dims[1], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], feature_dims[2], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(feature_dims[2], feature_dims[2], stride=1, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], feature_dims[2], 1, 1, 0)

    def _make_layer(self, in_dim, out_dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, norm_layer=norm_layer, stride=stride, dilation=dilation),
            ResidualBlock(out_dim, out_dim, norm_layer=norm_layer, stride=1, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer0(x)
        features.append(x)  # 1/1: 32, 256x256
        x = self.layer1(x)
        features.append(x)  # 1/2: 64, 128x128
        x = self.layer2(x)
        features.append(x)  # 1/4: 128, 64x64
        x = self.conv2(self.layer3(x))
        features.append(x)  # 1/4: 128, 64x64
        return features


class ResUnetDecoder(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d, feature_dims=(32, 64, 128)) -> None:
        super().__init__()
        self.decode_layer1 = self._make_fine_layer(2 * feature_dims[-1], feature_dims[-1], feature_dims[-2], norm_layer)
        self.decode_layer0 = self._make_fine_layer(2 * feature_dims[-2], feature_dims[-2], feature_dims[-3], norm_layer)
        self.out_layer0 = self._make_fine_layer(2 * feature_dims[0], feature_dims[0], feature_dims[0], norm_layer)
        self.out_layer1 = self._make_fine_layer(2 * feature_dims[1], feature_dims[1], feature_dims[1], norm_layer)
        self.out_layer2 = self._make_fine_layer(2 * feature_dims[2], feature_dims[2], feature_dims[2], norm_layer)

    def _make_fine_layer(self, in_dim, mid_dim, out_dim, norm_layer):
        return nn.Sequential(
            ResidualBlock(in_dim, mid_dim, norm_layer=norm_layer),
            ResidualBlock(mid_dim, out_dim, norm_layer=norm_layer),
        )

    def forward(
        self,
        feature_list: list[torch.Tensor],
        dino_feature_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        out_feature = []
        x = feature_list[-1]
        if dino_feature_list is not None:
            x = x + dino_feature_list[0]
        x = torch.cat([x, feature_list[-2]], dim=1)
        out_feature.append(self.out_layer2(x))

        x = self.decode_layer1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if dino_feature_list is not None:
            x = x + dino_feature_list[1]
        x = torch.cat([x, feature_list[-3]], dim=1)
        out_feature.append(self.out_layer1(x))

        x = self.decode_layer0(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if dino_feature_list is not None:
            x = x + dino_feature_list[2]
        x = torch.cat([x, feature_list[-4]], dim=1)
        out_feature.append(self.out_layer0(x))

        return out_feature

    def forward_64(
        self,
        feature_list: list[torch.Tensor],
        dino_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = feature_list[-1]
        if dino_feature is not None:
            x = x + dino_feature
        return self.out_layer2(torch.cat([x, feature_list[-2]], dim=1))


class ResUnet(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        norm_layer=nn.InstanceNorm2d,
        feature_dims=(32, 64, 128),
    ) -> None:
        super().__init__()
        self.encoder = ResUnetEncoder(norm_layer=norm_layer, feature_dims=feature_dims)
        self.decoder = ResUnetDecoder(norm_layer=norm_layer, feature_dims=feature_dims)
        self.up_dino_cnn = nn.ModuleList(
            [
                nn.Conv2d(dino_dim, feature_dims[-1], 1, bias=False),
                nn.Conv2d(dino_dim, feature_dims[-2], 1, bias=False),
                nn.Conv2d(dino_dim, feature_dims[-3], 1, bias=False),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        dino_feature: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feature_list = self.encoder(x)
        image_only_features = self.decoder(feature_list)
        dino_feature_list = []
        for i in range(len(self.up_dino_cnn)):
            dino_feature_i = self.up_dino_cnn[i](
                F.interpolate(
                    dino_feature,
                    size=feature_list[len(feature_list) - i - 2].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            dino_feature_list.append(dino_feature_i)
        fused_features = self.decoder(feature_list, dino_feature_list)
        return fused_features, image_only_features

    def forward_64(
        self,
        x: torch.Tensor,
        dino_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_list = self.encoder(x)
        image_only = self.decoder.forward_64(feature_list)
        dino_64 = self.up_dino_cnn[0](
            F.interpolate(
                dino_feature,
                size=feature_list[-2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        fused = self.decoder.forward_64(feature_list, dino_64)
        return fused, image_only


class ImageNetResUnetFeatureExtractor(nn.Module):
    def __init__(
        self,
        norm_layer=nn.InstanceNorm2d,
        feature_dims: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.encoder = ResUnetEncoder(norm_layer=norm_layer, feature_dims=feature_dims)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v = images.shape[:2]
        images = rearrange(images, "b v c h w -> (b v) c h w")
        images = (images.clamp(0, 1) - self.image_mean) / self.image_std

        encoder_features = self.encoder(images)
        return {
            "1_1": rearrange(encoder_features[0], "(b v) c h w -> b v c h w", b=b, v=v),
            "1_2": rearrange(encoder_features[1], "(b v) c h w -> b v c h w", b=b, v=v),
            "1_4": rearrange(encoder_features[2], "(b v) c h w -> b v c h w", b=b, v=v),
            "mid_1_4": rearrange(encoder_features[3], "(b v) c h w -> b v c h w", b=b, v=v),
        }


class HiSplatResUnetTokenFusion(nn.Module):
    def __init__(
        self,
        token_dim: int,
        token_ch: int = 64,
        feature_dims: tuple[int, int, int] = (32, 64, 128),
        norm_layer=nn.InstanceNorm2d,
        use_multiview_transformer: bool = False,
        multiview_transformer_layers: int = 6,
        multiview_transformer_heads: int = 1,
        multiview_attn_splits: int = 2,
    ) -> None:
        super().__init__()
        self.resunet = ResUnet(dino_dim=token_ch, norm_layer=norm_layer, feature_dims=feature_dims)
        self.use_multiview_transformer = use_multiview_transformer
        self.multiview_attn_splits = multiview_attn_splits
        self.multiview_transformer = (
            MultiViewFeatureTransformer(
                num_layers=multiview_transformer_layers,
                d_model=feature_dims[-1],
                nhead=multiview_transformer_heads,
                ffn_dim_expansion=4,
                no_cross_attn=False,
            )
            if use_multiview_transformer
            else None
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(token_dim, token_ch * 4, 1),
            nn.BatchNorm2d(token_ch * 4),
            nn.SiLU(),
        )
        self.upsampler0 = nn.Sequential(
            nn.ConvTranspose2d(token_ch * 4, token_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(token_ch * 2),
            nn.SiLU(),
        )
        self.upsampler1 = nn.Sequential(
            nn.ConvTranspose2d(token_ch * 2, token_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(token_ch),
            nn.SiLU(),
        )

    def load_unimatch_encoder(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"UniMatch checkpoint not found: {path}")

        print(f"==> Load compatible UniMatch weights into ResUNet encoder: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        pretrained = checkpoint.get("model", checkpoint)
        encoder_state = self.resunet.encoder.state_dict()
        updated_state_dict = {}
        skipped_shape = {}

        for key, value in pretrained.items():
            if not key.startswith("backbone."):
                continue

            possible_key = ".".join(key.split(".")[1:])
            if possible_key not in encoder_state:
                continue
            if encoder_state[possible_key].shape != value.shape:
                skipped_shape[possible_key] = (
                    tuple(value.shape),
                    tuple(encoder_state[possible_key].shape),
                )
                continue
            updated_state_dict[possible_key] = value

        updated_state_dict = OrderedDict(updated_state_dict)
        self.resunet.encoder.load_state_dict(updated_state_dict, strict=False)
        print(
            f"==> Loaded {len(updated_state_dict)} UniMatch tensors into "
            f"ResUNet encoder; skipped {len(skipped_shape)} shape mismatches"
        )
        if updated_state_dict:
            print(f"==> Loaded ResUNet keys: {list(updated_state_dict)}")

    def tokens_to_16x16(self, tokens: torch.Tensor) -> torch.Tensor:
        b, v, n, c = tokens.shape
        grid = int(math.sqrt(n))
        if grid * grid != n:
            raise ValueError(f"Expected square token grid, got {n} tokens.")
        token_map = rearrange(tokens, "b v (h w) c -> (b v) c h w", h=grid, w=grid)
        token_map = self.proj(token_map)
        token_map = self.upsampler0(token_map)
        token_map = self.upsampler1(token_map)
        if token_map.shape[-2:] != (16, 16):
            token_map = F.interpolate(token_map, size=(16, 16), mode="bilinear", align_corners=False)
        return token_map

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v = images.shape[:2]
        images = rearrange(images, "b v c h w -> (b v) c h w")
        images = (images.clamp(0, 1) - self.image_mean) / self.image_std
        dino_feature = self.tokens_to_16x16(tokens)
        fused_features, image_only_features = self.resunet(images, dino_feature)
        if self.multiview_transformer is not None:
            feature_64_list = [
                fused_features[0].reshape(b, v, *fused_features[0].shape[-3:])[:, i]
                for i in range(v)
            ]
            feature_64_list = feature_add_position_list(
                feature_64_list,
                self.multiview_attn_splits,
                fused_features[0].shape[1],
            )
            feature_64_list = self.multiview_transformer(
                feature_64_list,
                self.multiview_attn_splits,
            )
            fused_features[0] = rearrange(
                torch.stack(feature_64_list, dim=1),
                "b v c h w -> (b v) c h w",
            )
        return {
            "64": rearrange(fused_features[0], "(b v) c h w -> b v c h w", b=b, v=v),
            "128": rearrange(fused_features[1], "(b v) c h w -> b v c h w", b=b, v=v),
            "256": rearrange(fused_features[2], "(b v) c h w -> b v c h w", b=b, v=v),
            "image_only_64": rearrange(
                image_only_features[0],
                "(b v) c h w -> b v c h w",
                b=b,
                v=v,
            ),
            "image_only_128": rearrange(
                image_only_features[1],
                "(b v) c h w -> b v c h w",
                b=b,
                v=v,
            ),
            "image_only_256": rearrange(
                image_only_features[2],
                "(b v) c h w -> b v c h w",
                b=b,
                v=v,
            ),
        }
