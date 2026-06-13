import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..types import Gaussians
from .common.gaussians import build_covariance


def _make_offset_head(
    token_dim: int,
    hidden_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(token_dim),
        nn.Linear(token_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 3),
    )


def _make_attribute_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianResidualRefiner(nn.Module):
    """Decode LR/SR Gaussians in two stages after joint LitePT interaction."""

    def __init__(
        self,
        token_dim: int,
        sh_degree: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.sh_dim = (sh_degree + 1) ** 2
        self.attribute_dim = 3 + 4 + 1 + 3 * self.sh_dim
        attribute_input_dim = token_dim + 3

        self.lr_offset_head = _make_offset_head(token_dim, hidden_dim)
        self.sr_offset_head = _make_offset_head(token_dim, hidden_dim)
        self.lr_attribute_head = _make_attribute_head(
            attribute_input_dim,
            hidden_dim,
            self.attribute_dim,
        )
        self.sr_attribute_head = _make_attribute_head(
            attribute_input_dim,
            hidden_dim,
            self.attribute_dim,
        )

        for head in (
            self.lr_offset_head,
            self.sr_offset_head,
            self.lr_attribute_head,
            self.sr_attribute_head,
        ):
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-5)
            nn.init.zeros_(head[-1].bias)

    def _refine_branch_attributes(
        self,
        tokens: torch.Tensor,
        means: torch.Tensor,
        mean_offsets: torch.Tensor,
        gaussian_parameters: torch.Tensor,
        attribute_head: nn.Module,
    ) -> Gaussians:
        expected_parameter_dim = self.attribute_dim
        if gaussian_parameters.shape[-1] != expected_parameter_dim:
            raise ValueError(
                "Unexpected Gaussian state dimension: expected "
                f"{expected_parameter_dim}, got {gaussian_parameters.shape[-1]}."
            )

        cursor = 0
        initial_scales = gaussian_parameters[..., cursor : cursor + 3]
        cursor += 3
        initial_rotations = gaussian_parameters[..., cursor : cursor + 4]
        cursor += 4
        initial_opacity_logits = gaussian_parameters[
            ..., cursor : cursor + 1
        ].squeeze(-1)
        cursor += 1
        initial_sh = gaussian_parameters[..., cursor:]

        attribute_input = torch.cat(
            (tokens, mean_offsets),
            dim=-1,
        )
        residual = attribute_head(attribute_input)
        cursor = 0
        delta_scale = residual[..., cursor : cursor + 3]
        cursor += 3
        delta_rotation = residual[..., cursor : cursor + 4]
        cursor += 4
        delta_opacity = residual[..., cursor : cursor + 1].squeeze(-1)
        cursor += 1
        delta_sh = residual[..., cursor:]

        # Keep the existing Gaussian update rules unchanged for this iteration.
        scales = (initial_scales + delta_scale).clamp_min(1e-6)
        rotations = F.normalize(
            initial_rotations + delta_rotation,
            dim=-1,
        )
        opacities = torch.sigmoid(initial_opacity_logits + delta_opacity)
        harmonics = rearrange(
            initial_sh + delta_sh,
            "b n (rgb sh) -> b n rgb sh",
            rgb=3,
            sh=self.sh_dim,
        )
        return Gaussians(
            means=means,
            covariances=build_covariance(scales, rotations),
            rotations=rotations,
            scales=scales,
            harmonics=harmonics,
            opacities=opacities,
        )

    @staticmethod
    def _concatenate(first: Gaussians, second: Gaussians) -> Gaussians:
        return Gaussians(
            means=torch.cat((first.means, second.means), dim=1),
            covariances=torch.cat(
                (first.covariances, second.covariances),
                dim=1,
            ),
            rotations=torch.cat((first.rotations, second.rotations), dim=1),
            scales=torch.cat((first.scales, second.scales), dim=1),
            harmonics=torch.cat((first.harmonics, second.harmonics), dim=1),
            opacities=torch.cat((first.opacities, second.opacities), dim=1),
        )

    def decode_means(
        self,
        tokens: torch.Tensor,
        initial_means: torch.Tensor,
        num_lr_gaussians: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points = tokens.shape[1]
        if not 0 < num_lr_gaussians < num_points:
            raise ValueError(
                "num_lr_gaussians must split a non-empty LR and SR point set, "
                f"got {num_lr_gaussians} for {num_points} points."
            )
        if initial_means.shape[:2] != tokens.shape[:2]:
            raise ValueError(
                "Tokens and initial means must share their batch and point "
                "dimensions."
            )

        lr_slice = slice(0, num_lr_gaussians)
        sr_slice = slice(num_lr_gaussians, num_points)
        lr_offsets = self.lr_offset_head(tokens[:, lr_slice])
        sr_offsets = self.sr_offset_head(tokens[:, sr_slice])
        mean_offsets = torch.cat((lr_offsets, sr_offsets), dim=1)
        return initial_means + mean_offsets, mean_offsets

    def forward(
        self,
        tokens: torch.Tensor,
        refined_means: torch.Tensor,
        mean_offsets: torch.Tensor,
        gaussian_parameters: torch.Tensor,
        num_lr_gaussians: int,
    ) -> Gaussians:
        num_points = tokens.shape[1]
        if not 0 < num_lr_gaussians < num_points:
            raise ValueError(
                "num_lr_gaussians must split a non-empty LR and SR point set, "
                f"got {num_lr_gaussians} for {num_points} points."
            )
        if (
            refined_means.shape[:2] != tokens.shape[:2]
            or mean_offsets.shape != refined_means.shape
            or gaussian_parameters.shape[:2] != tokens.shape[:2]
        ):
            raise ValueError(
                "Tokens, refined means, mean offsets, and Gaussian states "
                "must share their batch and point dimensions."
            )

        lr_slice = slice(0, num_lr_gaussians)
        sr_slice = slice(num_lr_gaussians, num_points)
        lr_gaussians = self._refine_branch_attributes(
            tokens[:, lr_slice],
            refined_means[:, lr_slice],
            mean_offsets[:, lr_slice],
            gaussian_parameters[:, lr_slice],
            self.lr_attribute_head,
        )
        sr_gaussians = self._refine_branch_attributes(
            tokens[:, sr_slice],
            refined_means[:, sr_slice],
            mean_offsets[:, sr_slice],
            gaussian_parameters[:, sr_slice],
            self.sr_attribute_head,
        )
        return self._concatenate(lr_gaussians, sr_gaussians)
