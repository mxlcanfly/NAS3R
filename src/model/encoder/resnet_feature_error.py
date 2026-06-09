from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet18


class ResNet18FeatureErrorEncoder(nn.Module):
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"ResNet-18 weights not found: {weights_path}")

        backbone = resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(state_dict, strict=True)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

        for module in (self.conv1, self.bn1, self.layer1, self.layer2):
            module.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.conv1.eval()
        self.bn1.eval()
        self.layer1.eval()
        self.layer2.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, v, _, h, w = images.shape
        x = rearrange(images, "b v c h w -> (b v) c h w")
        x = (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

        with torch.no_grad():
            conv1 = self.conv1(x)
            x = self.maxpool(self.relu(self.bn1(conv1)))
            layer1 = self.layer1(x)
            layer2 = self.layer2(layer1)

        features = [
            F.interpolate(
                feature,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )
            for feature in (conv1, layer1, layer2)
        ]
        features = torch.cat(features, dim=1)
        return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
