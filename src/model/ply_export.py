from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    save_sh_dc_only: bool = False,
    as_text: bool = False,
    max_sh_degree: int | None = None,
    view_transform: bool = False,
):
    # Prune by opacity.
    mask = opacities >= 0.005
    opacities = opacities[mask]
    opacities, indices = torch.sort(opacities, descending=True)
    means = means[mask][indices]
    rotations = rotations[mask][indices]
    scales = scales[mask][indices]
    harmonics = harmonics[mask][indices]

    if view_transform and means.numel() > 0:
        # Viewer-only transform: center, scale, and rotate the scene so it opens
        # cleanly in SuperSplat. Disable this when raw world coordinates matter.
        means = means - means.median(dim=0).values
        scale_factor = means.abs().quantile(0.95, dim=0).max().clamp_min(1e-6)
        means = means / scale_factor
        scales = scales / scale_factor

        rotation = [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
        rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)
        adjustment = torch.tensor(
            R.from_rotvec([0, 0, -45], True).as_matrix(),
            dtype=torch.float32,
            device=means.device,
        )
        rotation = adjustment @ rotation
        rotation = rotation @ extrinsics[:3, :3].inverse()
        means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    if view_transform and means.size(0) > 0:
        rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    f_dc = harmonics[..., 0]
    if max_sh_degree is not None:
        max_coefficients = (max_sh_degree + 1) ** 2
        harmonics = harmonics[..., :max_coefficients]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(
            0 if save_sh_dc_only else f_rest.shape[1])
    ]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    if save_sh_dc_only:
        attributes = (
            means.detach().cpu().numpy(),
            torch.zeros_like(means).detach().cpu().numpy(),
            f_dc.detach().cpu().contiguous().numpy(),
            opacities[..., None].detach().cpu().numpy(),
            scales.log().detach().cpu().numpy(),
            rotations,
        )
    else:
        attributes = (
            means.detach().cpu().numpy(),
            torch.zeros_like(means).detach().cpu().numpy(),
            f_dc.detach().cpu().contiguous().numpy(),
            f_rest.detach().cpu().contiguous().numpy(),
            opacities[..., None].detach().cpu().numpy(),
            scales.log().detach().cpu().numpy(),
            rotations,
        )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=as_text).write(path)


def export_parent_child_debug_gaussians(
    extrinsics: Float[Tensor, "4 4"],
    parent_means: Float[Tensor, "parent 3"],
    child_means: Float[Tensor, "child 3"],
    parent_indices: Tensor,
    num_children: int,
    path: Path,
    as_text: bool = False,
    view_transform: bool = True,
) -> None:
    """Export selected parent-child centers as tiny colored Gaussian markers."""
    palette = torch.tensor(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
        ],
        dtype=parent_means.dtype,
        device=parent_means.device,
    ) / 255.0

    point_rows = []
    color_rows = []
    parent_indices_cpu = parent_indices.detach().long().cpu().tolist()
    for group_id, parent_idx in enumerate(parent_indices_cpu):
        if parent_idx < 0 or parent_idx >= parent_means.shape[0]:
            continue
        color = palette[group_id % len(palette)]
        point_rows.append(parent_means[parent_idx])
        color_rows.append(torch.ones(3, dtype=parent_means.dtype, device=parent_means.device))
        start = parent_idx * num_children
        end = start + num_children
        if start < 0 or end > child_means.shape[0]:
            continue
        for child_xyz in child_means[start:end]:
            point_rows.append(child_xyz)
            color_rows.append(color)

    if not point_rows:
        return

    means = torch.stack(point_rows).detach()
    colors = torch.stack(color_rows).detach()
    scene_scale = (means - means.median(dim=0).values).abs().quantile(0.95, dim=0).max().clamp_min(1e-6)
    marker_scale = scene_scale * 0.015
    scales = marker_scale.expand(means.shape[0], 3).clone()
    rotations = torch.zeros(means.shape[0], 4, dtype=means.dtype, device=means.device)
    rotations[:, 0] = 1.0
    opacities = torch.full((means.shape[0],), 0.95, dtype=means.dtype, device=means.device)
    sh_c0 = 0.28209479177387814
    f_dc = (colors - 0.5) / sh_c0
    harmonics = f_dc[:, :, None]

    export_ply(
        extrinsics,
        means,
        scales,
        rotations,
        harmonics,
        opacities,
        path,
        save_sh_dc_only=True,
        as_text=as_text,
        view_transform=view_transform,
    )
