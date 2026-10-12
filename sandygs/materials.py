"""Deterministic, view-consistent PBR microtextures for the sand surface.

The physical solver and Gaussian rasterizer define the macroscopic sand
surface.  This module only perturbs its per-pixel BRDF after depth and coverage
have been resolved; it must never change Gaussian opacity, depth or ownership.
All helpers except image loading are device-independent so the coordinate,
color and tangent-space contracts can be regression-tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        torch.pow((value + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=0.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * torch.pow(value, 1.0 / 2.4) - 0.055,
    )


def resolve_material_root(value: str | os.PathLike[str]) -> Path:
    """Expand a portable material root and reject unresolved variables."""

    raw = os.fspath(value)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if "$" in expanded:
        raise ValueError(
            f"sand texture root contains an unresolved environment variable: {raw}"
        )
    root = Path(expanded).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sand texture root is not a directory: {root}")
    return root


def _load_runtime_map(path: Path, expected_channels: int) -> torch.Tensor:
    """Load one runtime map as a normalized CPU ``[C,H,W]`` tensor."""

    # OpenCV is already a renderer dependency, but import it lazily so pure
    # material-math tests do not require image codecs during collection.
    import cv2
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"missing sand texture map: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to decode sand texture map: {path}")
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"sand texture map must be HxW or HxWxC: {path}")
    if expected_channels == 3:
        if image.shape[2] != 3:
            raise ValueError(
                f"sand texture map must have exactly three channels: {path}"
            )
        # OpenCV decodes color images as BGR; runtime normal maps use the same
        # stored RGB channel order as albedo and must be reversed as well.
        image = image[:, :, ::-1].copy()
    elif expected_channels == 1:
        if image.shape[2] != 1:
            raise ValueError(
                f"sand texture map must have exactly one channel: {path}"
            )
    else:
        raise ValueError("expected_channels must be one or three")
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise ValueError(
            f"sand texture map must use uint8 or uint16 storage: {path}"
        )
    tensor = torch.from_numpy(
        image.astype(np.float32, copy=False)
    ).permute(2, 0, 1) / scale
    if not torch.isfinite(tensor).all():
        raise ValueError(f"sand texture map contains non-finite values: {path}")
    return tensor.contiguous()


def decode_tangent_normal(encoded: torch.Tensor) -> torch.Tensor:
    """Decode an RGB normal map to normalized tangent-space vectors."""

    if encoded.ndim != 3 or encoded.shape[0] != 3:
        raise ValueError("encoded tangent normal must have shape [3,H,W]")
    if not torch.isfinite(encoded).all():
        raise ValueError("encoded tangent normal must contain only finite values")
    return F.normalize(encoded * 2.0 - 1.0, dim=0, eps=1.0e-8)


def encode_tangent_normal(normal: torch.Tensor) -> torch.Tensor:
    """Encode normalized tangent-space vectors as an RGB image in [0,1]."""

    if normal.ndim != 3 or normal.shape[0] != 3:
        raise ValueError("tangent normal must have shape [3,H,W]")
    if not torch.isfinite(normal).all():
        raise ValueError("tangent normal must contain only finite values")
    return torch.clamp(F.normalize(normal, dim=0, eps=1.0e-8) * 0.5 + 0.5, 0.0, 1.0)


def periodic_gaussian_lowpass(
    value: torch.Tensor,
    *,
    sigma_texels: float,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Apply a separable Gaussian low pass to a tileable ``[C,H,W]`` map."""

    if value.ndim != 3 or value.shape[0] <= 0:
        raise ValueError("periodic Gaussian input must have shape [C,H,W]")
    sigma_texels = float(sigma_texels)
    truncate = float(truncate)
    if not math.isfinite(sigma_texels) or sigma_texels <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if not math.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("Gaussian truncate must be finite and positive")
    if not torch.isfinite(value).all():
        raise ValueError("periodic Gaussian input must contain only finite values")

    radius = max(1, int(math.ceil(truncate * sigma_texels)))
    height, width = map(int, value.shape[1:])
    if radius > min(height, width):
        raise ValueError("Gaussian kernel radius must not exceed map dimensions")
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=value.device,
        dtype=value.dtype,
    )
    kernel = torch.exp(-0.5 * (coordinates / sigma_texels) ** 2)
    kernel = kernel / torch.clamp(kernel.sum(), min=1.0e-8)
    channels = int(value.shape[0])
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    lowpass = F.conv2d(
        F.pad(value[None], (radius, radius, 0, 0), mode="circular"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(lowpass, (0, 0, radius, radius), mode="circular"),
        vertical,
        groups=channels,
    )[0]


def extract_ambient_occlusion_microdetail(
    ambient_occlusion: torch.Tensor,
    *,
    sigma_texels: float,
    truncate: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Extract only darker-than-neighborhood AO detail from a tileable map.

    AO is linear: one means unoccluded and smaller values indicate crevices.
    The one-sided residual rejects broad illumination and bright embossing, so
    later shading can darken grain contacts without reintroducing macro waves.
    """

    if ambient_occlusion.ndim != 3 or ambient_occlusion.shape[0] != 1:
        raise ValueError("ambient occlusion must have shape [1,H,W]")
    if not torch.isfinite(ambient_occlusion).all():
        raise ValueError("ambient occlusion must contain only finite values")
    ambient_occlusion = torch.clamp(ambient_occlusion, 0.0, 1.0)
    lowpass = periodic_gaussian_lowpass(
        ambient_occlusion,
        sigma_texels=sigma_texels,
        truncate=truncate,
    )
    dark_residual = torch.clamp(lowpass - ambient_occlusion, min=0.0)
    return {
        "raw": ambient_occlusion,
        "lowpass": torch.clamp(lowpass, 0.0, 1.0),
        "dark_residual": dark_residual,
    }


def ambient_occlusion_from_residual(
    dark_residual: torch.Tensor,
    *,
    strength: float,
    minimum: float = 0.4,
    response: str = "linear",
    sparse_low: float = 0.04,
    sparse_high: float = 0.08,
    sparse_contrast: float = 0.40,
) -> torch.Tensor:
    """Convert a one-sided dark AO residual into a bounded visibility factor.

    ``linear`` preserves the original T4.1i behavior. ``sparse_smoothstep``
    rejects weak residuals below ``sparse_low`` and smoothly promotes only the
    strongest grain contacts to ``sparse_contrast`` darkness. Keeping the
    threshold interval separate from contrast lets an appearance profile make
    grains darker without simultaneously turning low-amplitude noise into dirt.
    """

    if dark_residual.ndim != 3 or dark_residual.shape[0] != 1:
        raise ValueError("AO dark residual must have shape [1,H,W]")
    strength = float(strength)
    minimum = float(minimum)
    response = str(response)
    sparse_low = float(sparse_low)
    sparse_high = float(sparse_high)
    sparse_contrast = float(sparse_contrast)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("AO strength must be finite and non-negative")
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("AO minimum must lie in [0,1]")
    if response not in {"linear", "sparse_smoothstep"}:
        raise ValueError("AO response must be linear or sparse_smoothstep")
    if not (
        math.isfinite(sparse_low)
        and math.isfinite(sparse_high)
        and 0.0 <= sparse_low < sparse_high
    ):
        raise ValueError(
            "AO sparse thresholds must satisfy finite 0 <= low < high"
        )
    if not math.isfinite(sparse_contrast) or not 0.0 <= sparse_contrast <= 1.0:
        raise ValueError("AO sparse contrast must lie in [0,1]")
    if not torch.isfinite(dark_residual).all() or torch.any(dark_residual < 0.0):
        raise ValueError("AO dark residual must be finite and non-negative")
    if response == "linear":
        # Keep the exact original expression so existing AO profiles remain
        # pixel-compatible unless they opt into the sparse response explicitly.
        return torch.clamp(
            1.0 - strength * dark_residual, min=minimum, max=1.0
        )

    normalized = torch.clamp(
        (dark_residual - sparse_low) / (sparse_high - sparse_low),
        min=0.0,
        max=1.0,
    )
    smooth_weight = normalized * normalized * (3.0 - 2.0 * normalized)
    return torch.clamp(
        1.0 - sparse_contrast * smooth_weight,
        min=minimum,
        max=1.0,
    )


def highpass_tangent_normal(
    normal: torch.Tensor,
    *,
    sigma_texels: float,
    strength: float = 1.0,
    truncate: float = 3.0,
) -> torch.Tensor:
    """Remove low-frequency tangent slopes while retaining micro-normal detail.

    Filtering decoded XYZ channels directly biases their length and Z value.
    Instead, convert the normal to tangent-plane slopes, subtract a periodic
    Gaussian low pass, then reconstruct a unit vector around neutral +Z. The
    macroscopic sand normal remains supplied by the simulated triangle surface.
    Periodic padding preserves the repeatable texture seam.
    """

    if normal.ndim != 3 or normal.shape[0] != 3:
        raise ValueError("tangent normal must have shape [3,H,W]")
    sigma_texels = float(sigma_texels)
    strength = float(strength)
    truncate = float(truncate)
    if not math.isfinite(sigma_texels) or sigma_texels <= 0.0:
        raise ValueError("normal high-pass sigma must be finite and positive")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("normal high-pass strength must be finite and non-negative")
    if not math.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("normal high-pass truncate must be finite and positive")
    if not torch.isfinite(normal).all():
        raise ValueError("tangent normal must contain only finite values")

    normal = F.normalize(normal, dim=0, eps=1.0e-8)
    # Runtime Normal-GL maps describe the upper tangent hemisphere. Clamp only
    # the denominator so a malformed grazing texel cannot create infinite slope.
    slopes = normal[:2] / torch.clamp(normal[2:3], min=1.0e-4)
    lowpass = periodic_gaussian_lowpass(
        slopes,
        sigma_texels=sigma_texels,
        truncate=truncate,
    )
    micro_slopes = strength * (slopes - lowpass)
    return F.normalize(
        torch.cat((micro_slopes, torch.ones_like(micro_slopes[:1])), dim=0),
        dim=0,
        eps=1.0e-8,
    )


@dataclass(frozen=True)
class SandPBRMaterial:
    """GPU-ready mip pyramid and normalization statistics for one material."""

    root: Path
    # Each level packs linear albedo RGB, roughness, decoded tangent normal XYZ,
    # raw AO and its dark high-pass residual into one nine-channel tensor. AO-off
    # materials store neutral 1/0 channels and do not read or filter ``ao.jpg``.
    levels: tuple[torch.Tensor, ...]
    albedo_mean_linear: torch.Tensor
    roughness_mean: torch.Tensor
    base_height: int
    base_width: int

    @classmethod
    def load(
        cls,
        root: str | os.PathLike[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        normal_highpass_enabled: bool = False,
        normal_highpass_sigma_texels: float = 8.0,
        normal_highpass_strength: float = 1.0,
        ambient_occlusion_enabled: bool = False,
        ambient_occlusion_highpass_sigma_texels: float = 8.0,
    ) -> "SandPBRMaterial":
        runtime_root = resolve_material_root(root)
        albedo = _load_runtime_map(runtime_root / "albedo.jpg", 3)
        roughness = _load_runtime_map(runtime_root / "roughness.jpg", 1)
        normal_encoded = _load_runtime_map(runtime_root / "normal_gl.png", 3)
        if albedo.shape[1:] != roughness.shape[1:] or (
            albedo.shape[1:] != normal_encoded.shape[1:]
        ):
            raise ValueError("sand runtime maps must have identical dimensions")
        height, width = map(int, albedo.shape[1:])
        if height <= 0 or width <= 0:
            raise ValueError("sand runtime maps must be non-empty")

        albedo = _srgb_to_linear(albedo).to(device=device, dtype=dtype)
        roughness = roughness.to(device=device, dtype=dtype)
        normal = decode_tangent_normal(
            normal_encoded.to(device=device, dtype=dtype)
        )
        ambient_occlusion = torch.ones(
            (1, height, width), device=device, dtype=dtype
        )
        ambient_occlusion_dark_residual = torch.zeros_like(ambient_occlusion)
        if ambient_occlusion_enabled:
            ambient_occlusion_source = _load_runtime_map(
                runtime_root / "ao.jpg", 1
            )
            if ambient_occlusion_source.shape[1:] != (height, width):
                raise ValueError("sand runtime maps must have identical dimensions")
            ambient_occlusion_detail = extract_ambient_occlusion_microdetail(
                ambient_occlusion_source.to(device=device, dtype=dtype),
                sigma_texels=ambient_occlusion_highpass_sigma_texels,
            )
            ambient_occlusion = ambient_occlusion_detail["raw"]
            ambient_occlusion_dark_residual = ambient_occlusion_detail[
                "dark_residual"
            ]
        if normal_highpass_enabled:
            # Filter before mip construction so every later LOD is derived
            # from the same view-independent micro-normal field.
            normal = highpass_tangent_normal(
                normal,
                sigma_texels=normal_highpass_sigma_texels,
                strength=normal_highpass_strength,
            )
        albedo_mean = torch.clamp(
            albedo.mean(dim=(1, 2), keepdim=True), min=1.0e-4
        )
        roughness_mean = roughness.mean().reshape(1, 1, 1)

        levels: list[torch.Tensor] = []
        current_albedo = albedo[None]
        current_roughness = roughness[None]
        current_normal = normal[None]
        current_ambient_occlusion = ambient_occlusion[None]
        current_ambient_occlusion_dark_residual = (
            ambient_occlusion_dark_residual[None]
        )
        while True:
            levels.append(torch.cat(
                (
                    current_albedo,
                    current_roughness,
                    current_normal,
                    current_ambient_occlusion,
                    current_ambient_occlusion_dark_residual,
                ),
                dim=1,
            ))
            current_height, current_width = current_albedo.shape[-2:]
            if current_height == 1 and current_width == 1:
                break
            next_size = (
                max(1, current_height // 2),
                max(1, current_width // 2),
            )
            current_albedo = F.interpolate(
                current_albedo,
                size=next_size,
                mode="area",
            )
            current_roughness = F.interpolate(
                current_roughness,
                size=next_size,
                mode="area",
            )
            current_normal = F.normalize(
                F.interpolate(current_normal, size=next_size, mode="area"),
                dim=1,
                eps=1.0e-8,
            )
            current_ambient_occlusion = F.interpolate(
                current_ambient_occlusion, size=next_size, mode="area"
            )
            current_ambient_occlusion_dark_residual = F.interpolate(
                current_ambient_occlusion_dark_residual,
                size=next_size,
                mode="area",
            )
        return cls(
            root=runtime_root,
            levels=tuple(levels),
            albedo_mean_linear=albedo_mean,
            roughness_mean=roughness_mean,
            base_height=height,
            base_width=width,
        )

    @property
    def max_lod(self) -> int:
        return len(self.levels) - 1


def reconstruct_world_positions(
    camera_depth: torch.Tensor,
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    world_view_transform: torch.Tensor,
) -> torch.Tensor:
    """Back-project camera-Z depth and transform it to GS world coordinates.

    The camera representation stores a transposed world-to-view matrix and applies
    points as row vectors.  Consequently ``[p_camera,1] @ inverse(WV)`` is the
    exact inverse of the source/sand rasterization transform.
    """

    if camera_depth.ndim != 3 or camera_depth.shape[0] != 1:
        raise ValueError("camera_depth must have shape [1,H,W]")
    if world_view_transform.shape != (4, 4):
        raise ValueError("world_view_transform must have shape [4,4]")
    height, width = camera_depth.shape[1:]
    device, dtype = camera_depth.device, camera_depth.dtype
    fx_t = torch.as_tensor(fx, device=device, dtype=dtype)
    fy_t = torch.as_tensor(fy, device=device, dtype=dtype)
    cx_t = torch.as_tensor(cx, device=device, dtype=dtype)
    cy_t = torch.as_tensor(cy, device=device, dtype=dtype)
    if torch.any(fx_t <= 0.0) or torch.any(fy_t <= 0.0):
        raise ValueError("camera focal lengths must be positive")
    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    depth = camera_depth[0]
    camera = torch.stack(
        (
            (grid_u - cx_t) / fx_t * depth,
            (grid_v - cy_t) / fy_t * depth,
            depth,
        ),
        dim=-1,
    )
    homogeneous = torch.cat(
        (camera, torch.ones_like(camera[..., :1])), dim=-1
    )
    world_h = homogeneous @ torch.linalg.inv(world_view_transform)
    safe_w = torch.where(
        torch.abs(world_h[..., 3:4]) > 1.0e-8,
        world_h[..., 3:4],
        torch.ones_like(world_h[..., 3:4]),
    )
    return (world_h[..., :3] / safe_w).movedim(-1, 0)


def world_positions_to_simulation(
    positions_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
) -> torch.Tensor:
    """Transform ``[3,H,W]`` GS positions into simulation coordinates."""

    if positions_world.ndim != 3 or positions_world.shape[0] != 3:
        raise ValueError("positions_world must have shape [3,H,W]")
    if sim_to_gs.shape != (4, 4):
        raise ValueError("sim_to_gs must have shape [4,4]")
    points = positions_world.movedim(0, -1)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    # ``sim_to_gs`` follows the column-vector exchange convention.  Row-vector
    # point batches therefore multiply by the transpose of its inverse.
    gs_to_sim = torch.linalg.inv(sim_to_gs)
    simulation_h = homogeneous @ gs_to_sim.transpose(0, 1)
    safe_w = torch.where(
        torch.abs(simulation_h[..., 3:4]) > 1.0e-8,
        simulation_h[..., 3:4],
        torch.ones_like(simulation_h[..., 3:4]),
    )
    return (simulation_h[..., :3] / safe_w).movedim(-1, 0)


def simulation_xy_uv(
    positions_simulation: torch.Tensor,
    *,
    period: float,
    rotation_degrees: float = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
) -> torch.Tensor:
    """Return unwrapped, camera-independent UV coordinates as ``[2,H,W]``."""

    if positions_simulation.ndim != 3 or positions_simulation.shape[0] != 3:
        raise ValueError("positions_simulation must have shape [3,H,W]")
    period = float(period)
    rotation_degrees = float(rotation_degrees)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sand texture period must be finite and positive")
    if not math.isfinite(rotation_degrees):
        raise ValueError("sand texture rotation must be finite")
    if len(offset) != 2 or any(not math.isfinite(float(x)) for x in offset):
        raise ValueError("sand texture offset must contain two finite numbers")
    angle = math.radians(rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y = positions_simulation[0], positions_simulation[1]
    rotated_x = cosine * x - sine * y
    rotated_y = sine * x + cosine * y
    return torch.stack(
        (
            rotated_x / period + float(offset[0]),
            rotated_y / period + float(offset[1]),
        ),
        dim=0,
    )


def simulation_triplanar_uvs(
    positions_simulation: torch.Tensor,
    *,
    period: float,
    rotation_degrees: float = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
) -> dict[str, torch.Tensor]:
    """Build XY/XZ/YZ projections from fixed simulation-space positions.

    XY is produced by :func:`simulation_xy_uv` exactly, keeping horizontal
    sand pixel-compatible with the original projection when its triplanar
    weight is one. The same in-plane rotation and phase are used for both side
    projections so all channels remain locked to simulation space over time.
    """

    uv_xy = simulation_xy_uv(
        positions_simulation,
        period=period,
        rotation_degrees=rotation_degrees,
        offset=offset,
    )
    angle = math.radians(float(rotation_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    period = float(period)

    def project(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        rotated_first = cosine * first - sine * second
        rotated_second = sine * first + cosine * second
        return torch.stack(
            (
                rotated_first / period + float(offset[0]),
                rotated_second / period + float(offset[1]),
            ),
            dim=0,
        )

    x, y, z = positions_simulation
    return {
        "xy": uv_xy,
        "xz": project(x, z),
        "yz": project(y, z),
    }


def simulation_triplanar_weights(
    macro_normal_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
    *,
    sharpness: float = 4.0,
) -> torch.Tensor:
    """Return normalized XY/XZ/YZ projection weights as ``[3,H,W]``.

    Normals transform back to simulation coordinates with ``A^T`` for the
    column-vector point transform ``p_gs=A p_sim+t``. Absolute components make
    the choice two-sided: Z selects XY, Y selects XZ and X selects YZ.
    """

    if macro_normal_world.ndim != 3 or macro_normal_world.shape[0] != 3:
        raise ValueError("macro_normal_world must have shape [3,H,W]")
    if sim_to_gs.shape != (4, 4):
        raise ValueError("sim_to_gs must have shape [4,4]")
    sharpness = float(sharpness)
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("triplanar sharpness must be finite and positive")
    normal_world = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    linear = sim_to_gs[:3, :3]
    normal_simulation = (
        linear.transpose(0, 1) @ normal_world.reshape(3, -1)
    ).reshape_as(normal_world)
    normal_simulation = F.normalize(normal_simulation, dim=0, eps=1.0e-8)
    absolute = torch.abs(normal_simulation)
    unnormalized = torch.stack(
        (absolute[2], absolute[1], absolute[0]), dim=0
    ) ** sharpness
    return unnormalized / torch.clamp(
        unnormalized.sum(dim=0, keepdim=True), min=1.0e-8
    )


def estimate_texture_lod(
    uv: torch.Tensor,
    *,
    texture_height: int,
    texture_width: int,
    max_lod: int,
    bias: float = 0.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate a screen-space mip level from unwrapped UV derivatives."""

    if uv.ndim != 3 or uv.shape[0] != 2:
        raise ValueError("uv must have shape [2,H,W]")
    if texture_height <= 0 or texture_width <= 0 or max_lod < 0:
        raise ValueError("texture dimensions and max_lod must be valid")
    if valid is not None and valid.shape != (1,) + tuple(uv.shape[1:]):
        raise ValueError("valid must have shape [1,H,W] matching uv")
    bias = float(bias)
    if not math.isfinite(bias):
        raise ValueError("sand texture LOD bias must be finite")
    if uv.shape[2] > 1:
        du_dx = uv[:, :, 1:] - uv[:, :, :-1]
        if valid is not None:
            valid_dx = valid[:, :, 1:] & valid[:, :, :-1]
            du_dx = torch.where(valid_dx.repeat(2, 1, 1), du_dx, 0.0)
        du_dx = F.pad(du_dx, (0, 1, 0, 0), mode="replicate")
    else:
        du_dx = torch.zeros_like(uv)
    if uv.shape[1] > 1:
        du_dy = uv[:, 1:, :] - uv[:, :-1, :]
        if valid is not None:
            valid_dy = valid[:, 1:, :] & valid[:, :-1, :]
            du_dy = torch.where(valid_dy.repeat(2, 1, 1), du_dy, 0.0)
        du_dy = F.pad(du_dy, (0, 0, 0, 1), mode="replicate")
    else:
        du_dy = torch.zeros_like(uv)
    scale = torch.tensor(
        (float(texture_width), float(texture_height)),
        dtype=uv.dtype,
        device=uv.device,
    )[:, None, None]
    rho_x = torch.linalg.vector_norm(du_dx * scale, dim=0, keepdim=True)
    rho_y = torch.linalg.vector_norm(du_dy * scale, dim=0, keepdim=True)
    rho = torch.maximum(rho_x, rho_y)
    lod = torch.log2(torch.clamp(rho, min=1.0)) + bias
    return torch.clamp(lod, min=0.0, max=float(max_lod))


def _sample_repeating_level(level: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample one texture level with seamless repeat wrapping."""

    height, width = level.shape[-2:]
    padded = F.pad(level, (1, 1, 1, 1), mode="circular")
    wrapped = uv - torch.floor(uv)
    # UV follows the OpenGL convention (+V points up), while decoded image row
    # zero is the top.  The vertical flip also keeps Normal-GL's green channel
    # aligned with the positive simulation-Y tangent.
    x_pixel = wrapped[0] * width + 0.5
    y_pixel = (1.0 - wrapped[1]) * height + 0.5
    grid_x = 2.0 * (x_pixel + 0.5) / (width + 2.0) - 1.0
    grid_y = 2.0 * (y_pixel + 0.5) / (height + 2.0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    return F.grid_sample(
        padded,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]


def sample_sand_pbr_material(
    material: SandPBRMaterial,
    uv: torch.Tensor,
    valid: torch.Tensor,
    *,
    lod_bias: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Trilinearly sample packed PBR maps only over visible sand support."""

    if valid.shape != (1,) + tuple(uv.shape[1:]):
        raise ValueError("valid must have shape [1,H,W] matching uv")
    valid = valid.to(torch.bool)
    lod = estimate_texture_lod(
        uv,
        texture_height=material.base_height,
        texture_width=material.base_width,
        max_lod=material.max_lod,
        bias=lod_bias,
        valid=valid,
    )
    packed = torch.zeros(
        (9,) + tuple(uv.shape[1:]), dtype=uv.dtype, device=uv.device
    )
    if torch.any(valid):
        selected_lod = lod[valid]
        minimum_level = max(0, int(torch.floor(selected_lod.min()).item()))
        maximum_level = min(
            material.max_lod,
            int(torch.ceil(selected_lod.max()).item()),
        )
        valid_float = valid.to(uv.dtype)
        for level_index in range(minimum_level, maximum_level + 1):
            weight = torch.clamp(
                1.0 - torch.abs(lod - float(level_index)), 0.0, 1.0
            ) * valid_float
            if not torch.any(weight > 0.0):
                continue
            packed = packed + _sample_repeating_level(
                material.levels[level_index], uv
            ) * weight
    tangent_normal = F.normalize(packed[4:7], dim=0, eps=1.0e-8)
    default_normal = torch.zeros_like(tangent_normal)
    default_normal[2] = 1.0
    tangent_normal = torch.where(
        valid.repeat(3, 1, 1), tangent_normal, default_normal
    )
    return {
        "albedo_linear": packed[0:3],
        "roughness": packed[3:4],
        "normal_tangent": tangent_normal,
        "ambient_occlusion_raw": packed[7:8],
        "ambient_occlusion_dark_residual": packed[8:9],
        "lod": torch.where(valid, lod, torch.zeros_like(lod)),
    }


def build_simulation_tangent_basis(
    macro_normal_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project simulation +X/+Y axes into the macroscopic sand tangent plane."""

    if macro_normal_world.ndim != 3 or macro_normal_world.shape[0] != 3:
        raise ValueError("macro_normal_world must have shape [3,H,W]")
    if sim_to_gs.shape != (4, 4):
        raise ValueError("sim_to_gs must have shape [4,4]")
    normal = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    linear = sim_to_gs[:3, :3]
    axis_x = linear[:, 0]
    axis_y = linear[:, 1]
    axis_x = axis_x / torch.clamp(torch.linalg.vector_norm(axis_x), min=1.0e-8)
    axis_y = axis_y / torch.clamp(torch.linalg.vector_norm(axis_y), min=1.0e-8)
    tangent = axis_x[:, None, None] - normal * torch.sum(
        normal * axis_x[:, None, None], dim=0, keepdim=True
    )
    tangent_norm = torch.linalg.vector_norm(tangent, dim=0, keepdim=True)
    fallback_tangent = axis_y[:, None, None] - normal * torch.sum(
        normal * axis_y[:, None, None], dim=0, keepdim=True
    )
    tangent = torch.where(
        (tangent_norm > 1.0e-6).repeat(3, 1, 1), tangent, fallback_tangent
    )
    tangent = F.normalize(tangent, dim=0, eps=1.0e-8)
    bitangent = axis_y[:, None, None] - normal * torch.sum(
        normal * axis_y[:, None, None], dim=0, keepdim=True
    )
    bitangent = bitangent - tangent * torch.sum(
        bitangent * tangent, dim=0, keepdim=True
    )
    bitangent_norm = torch.linalg.vector_norm(bitangent, dim=0, keepdim=True)
    fallback_bitangent = torch.linalg.cross(normal, tangent, dim=0)
    bitangent = torch.where(
        (bitangent_norm > 1.0e-6).repeat(3, 1, 1),
        bitangent,
        fallback_bitangent,
    )
    return tangent, F.normalize(bitangent, dim=0, eps=1.0e-8)


def _build_triplanar_tangent_basis(
    macro_normal_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
    plane_axes: tuple[int, int],
    rotation_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one rotated simulation plane into the sand tangent plane."""

    normal = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    linear = sim_to_gs[:3, :3]
    first = F.normalize(linear[:, plane_axes[0]], dim=0, eps=1.0e-8)
    second = F.normalize(linear[:, plane_axes[1]], dim=0, eps=1.0e-8)
    angle = math.radians(float(rotation_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    # UV = R * [first_coordinate, second_coordinate]. The corresponding
    # increasing-U/V directions are the columns of R^T in the physical plane.
    axis_u = cosine * first - sine * second
    axis_v = sine * first + cosine * second
    projected_u = axis_u[:, None, None] - normal * torch.sum(
        normal * axis_u[:, None, None], dim=0, keepdim=True
    )
    projected_v = axis_v[:, None, None] - normal * torch.sum(
        normal * axis_v[:, None, None], dim=0, keepdim=True
    )
    use_u = torch.linalg.vector_norm(
        projected_u, dim=0, keepdim=True
    ) > 1.0e-6
    tangent = torch.where(use_u.repeat(3, 1, 1), projected_u, projected_v)
    secondary = torch.where(
        use_u.repeat(3, 1, 1), projected_v, projected_u
    )
    tangent = F.normalize(tangent, dim=0, eps=1.0e-8)
    bitangent = secondary - tangent * torch.sum(
        secondary * tangent, dim=0, keepdim=True
    )
    bitangent_norm = torch.linalg.vector_norm(
        bitangent, dim=0, keepdim=True
    )
    fallback = torch.linalg.cross(normal, tangent, dim=0)
    bitangent = torch.where(
        (bitangent_norm > 1.0e-6).repeat(3, 1, 1),
        bitangent,
        fallback,
    )
    return tangent, F.normalize(bitangent, dim=0, eps=1.0e-8)


def _triplanar_tangent_normal_to_world(
    tangent_normal: torch.Tensor,
    macro_normal_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
    plane_axes: tuple[int, int],
    *,
    rotation_degrees: float,
    normal_strength: float,
) -> torch.Tensor:
    """Reorient one projected normal map around the true macro surface.

    The map's +Z component is applied along the simulated sand normal, not the
    projection axis. Consequently a neutral normal map always reproduces the
    macro normal for every plane and every blend sharpness.
    """

    tangent_normal = tangent_normal.clone()
    tangent_normal[:2] *= float(normal_strength)
    tangent_normal = F.normalize(tangent_normal, dim=0, eps=1.0e-8)
    tangent, bitangent = _build_triplanar_tangent_basis(
        macro_normal_world,
        sim_to_gs,
        plane_axes,
        rotation_degrees,
    )
    macro_normal = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    return F.normalize(
        tangent * tangent_normal[0:1]
        + bitangent * tangent_normal[1:2]
        + macro_normal * tangent_normal[2:3],
        dim=0,
        eps=1.0e-8,
    )


def sample_sand_pbr_material_triplanar(
    material: SandPBRMaterial,
    positions_simulation: torch.Tensor,
    macro_normal_world: torch.Tensor,
    sim_to_gs: torch.Tensor,
    valid: torch.Tensor,
    *,
    period: float,
    rotation_degrees: float = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
    sharpness: float = 4.0,
    lod_bias: float = 0.0,
    normal_strength: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sample and blend XY/XZ/YZ material projections over visible sand.

    Each plane computes its own screen-space LOD, removing the XY derivative
    degeneracy on steep faces. Scalar/color channels blend directly. Normals
    are reoriented into the macro surface's world-space tangent plane before
    blending; raw tangent normals from different projections are never treated
    as though they shared one TBN frame.
    """

    if positions_simulation.ndim != 3 or positions_simulation.shape[0] != 3:
        raise ValueError("positions_simulation must have shape [3,H,W]")
    expected_scalar = (1,) + tuple(positions_simulation.shape[1:])
    if valid.shape != expected_scalar:
        raise ValueError("valid must have shape [1,H,W]")
    if macro_normal_world.shape != positions_simulation.shape:
        raise ValueError("macro normal and simulation positions must match")
    normal_strength = float(normal_strength)
    if not math.isfinite(normal_strength) or normal_strength < 0.0:
        raise ValueError("sand texture normal strength must be finite and non-negative")

    positions_simulation = torch.nan_to_num(
        positions_simulation, nan=0.0, posinf=0.0, neginf=0.0
    )
    uvs = simulation_triplanar_uvs(
        positions_simulation,
        period=period,
        rotation_degrees=rotation_degrees,
        offset=offset,
    )
    weights = simulation_triplanar_weights(
        macro_normal_world,
        sim_to_gs,
        sharpness=sharpness,
    )
    plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    height, width = positions_simulation.shape[1:]
    scalar_shape = (1, height, width)
    vector_shape = (3, height, width)
    blended = {
        "albedo_linear": torch.zeros(
            vector_shape,
            dtype=positions_simulation.dtype,
            device=positions_simulation.device,
        ),
        "roughness": torch.zeros(
            scalar_shape,
            dtype=positions_simulation.dtype,
            device=positions_simulation.device,
        ),
        "ambient_occlusion_raw": torch.zeros(
            scalar_shape,
            dtype=positions_simulation.dtype,
            device=positions_simulation.device,
        ),
        "ambient_occlusion_dark_residual": torch.zeros(
            scalar_shape,
            dtype=positions_simulation.dtype,
            device=positions_simulation.device,
        ),
        "lod": torch.zeros(
            scalar_shape,
            dtype=positions_simulation.dtype,
            device=positions_simulation.device,
        ),
    }
    tangent_normal_sum = torch.zeros_like(blended["albedo_linear"])
    world_normal_sum = torch.zeros_like(blended["albedo_linear"])
    checker_sum = torch.zeros_like(blended["albedo_linear"])
    plane_lods: list[torch.Tensor] = []
    for plane_index, plane_name in enumerate(("xy", "xz", "yz")):
        uv = uvs[plane_name]
        sampled = sample_sand_pbr_material(
            material,
            uv,
            valid,
            lod_bias=lod_bias,
        )
        weight = weights[plane_index : plane_index + 1]
        blended["albedo_linear"] += sampled["albedo_linear"] * weight
        blended["roughness"] += sampled["roughness"] * weight
        blended["ambient_occlusion_raw"] += (
            sampled["ambient_occlusion_raw"] * weight
        )
        blended["ambient_occlusion_dark_residual"] += (
            sampled["ambient_occlusion_dark_residual"] * weight
        )
        blended["lod"] += sampled["lod"] * weight
        tangent_normal_sum += sampled["normal_tangent"] * weight
        world_normal_sum += _triplanar_tangent_normal_to_world(
            sampled["normal_tangent"],
            macro_normal_world,
            sim_to_gs,
            plane_axes[plane_name],
            rotation_degrees=rotation_degrees,
            normal_strength=normal_strength,
        ) * weight
        checker_sum += sand_texture_uv_checker(uv, valid) * weight
        plane_lods.append(sampled["lod"])

    blended["normal_tangent"] = F.normalize(
        tangent_normal_sum, dim=0, eps=1.0e-8
    )
    blended["normal_world_override"] = F.normalize(
        world_normal_sum, dim=0, eps=1.0e-8
    )
    blended["triplanar_weights"] = (
        weights * valid.to(weights.dtype)
    )
    blended["triplanar_lod"] = torch.cat(plane_lods, dim=0)
    blended["uv_checker"] = checker_sum
    return blended


def apply_sand_pbr_microtexture(
    base_albedo_srgb: torch.Tensor,
    base_roughness: torch.Tensor,
    macro_normal_world: torch.Tensor,
    sampled: dict[str, torch.Tensor],
    material: SandPBRMaterial,
    sim_to_gs: torch.Tensor,
    valid: torch.Tensor,
    *,
    albedo_strength: float,
    roughness_strength: float,
    normal_strength: float,
    albedo_detail_min: float = 0.5,
    albedo_detail_max: float = 1.5,
    roughness_min: float = 0.08,
    roughness_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply zero-mean material detail without changing geometry or opacity."""

    if base_albedo_srgb.ndim != 3 or base_albedo_srgb.shape[0] != 3:
        raise ValueError("base_albedo_srgb must have shape [3,H,W]")
    if base_roughness.shape != (1,) + tuple(base_albedo_srgb.shape[1:]):
        raise ValueError("base_roughness must have shape [1,H,W]")
    if valid.shape != base_roughness.shape:
        raise ValueError("valid must match base_roughness")
    strengths = (albedo_strength, roughness_strength, normal_strength)
    if any(not math.isfinite(float(x)) or float(x) < 0.0 for x in strengths):
        raise ValueError("sand texture strengths must be finite and non-negative")
    if not 0.0 < albedo_detail_min <= albedo_detail_max:
        raise ValueError("sand albedo detail bounds must be positive and ordered")
    if not 0.0 <= roughness_min <= roughness_max <= 1.0:
        raise ValueError("sand roughness bounds must lie in [0,1]")

    detail = torch.clamp(
        sampled["albedo_linear"] / material.albedo_mean_linear,
        min=float(albedo_detail_min),
        max=float(albedo_detail_max),
    )
    albedo_factor = torch.lerp(
        torch.ones_like(detail), detail, float(albedo_strength)
    )
    textured_albedo = _linear_to_srgb(
        _srgb_to_linear(base_albedo_srgb) * albedo_factor
    )
    textured_roughness = torch.clamp(
        base_roughness
        + float(roughness_strength)
        * (sampled["roughness"] - material.roughness_mean),
        min=float(roughness_min),
        max=float(roughness_max),
    )
    tangent_normal = sampled["normal_tangent"].clone()
    tangent_normal[:2] *= float(normal_strength)
    tangent_normal = F.normalize(tangent_normal, dim=0, eps=1.0e-8)
    macro_normal = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    if "normal_world_override" in sampled:
        # Triplanar samples have already been reoriented per projection before
        # blending; applying the legacy XY TBN again would rotate them twice.
        shading_normal = F.normalize(
            sampled["normal_world_override"], dim=0, eps=1.0e-8
        )
    else:
        tangent, bitangent = build_simulation_tangent_basis(
            macro_normal_world, sim_to_gs
        )
        shading_normal = F.normalize(
            tangent * tangent_normal[0:1]
            + bitangent * tangent_normal[1:2]
            + macro_normal * tangent_normal[2:3],
            dim=0,
            eps=1.0e-8,
        )
    valid_rgb = valid.to(torch.bool).repeat(3, 1, 1)
    return {
        "albedo_srgb": torch.where(
            valid_rgb, textured_albedo, base_albedo_srgb
        ),
        "albedo_factor": torch.where(
            valid_rgb, albedo_factor, torch.ones_like(albedo_factor)
        ),
        "roughness": torch.where(
            valid.to(torch.bool), textured_roughness, base_roughness
        ),
        "normal_tangent": tangent_normal,
        "normal_world": torch.where(
            valid_rgb, shading_normal, macro_normal_world
        ),
    }


def apply_sand_texture_primary(
    base_albedo_srgb: torch.Tensor,
    base_roughness: torch.Tensor,
    macro_normal_world: torch.Tensor,
    sampled: dict[str, torch.Tensor],
    sim_to_gs: torch.Tensor,
    valid: torch.Tensor,
    *,
    albedo_gain: Sequence[float] = (1.0, 1.0, 1.0),
    roughness_scale: float = 1.0,
    normal_strength: float = 1.0,
    roughness_min: float = 0.08,
    roughness_max: float = 1.0,
    wetness: torch.Tensor | None = None,
    wet_adjustment_enabled: bool = False,
    wet_albedo_scale: float = 0.65,
    wet_response_rate: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Use sampled PBR maps as the sand material endpoint.

    Unlike :func:`apply_sand_pbr_microtexture`, this mode does not neutralize
    the texture around an artist-authored dry/wet base.  Valid sand pixels use
    the sampled linear Albedo and Roughness directly (modulo explicit unit
    gains), while invalid pixels retain the pre-existing material.  The normal
    map remains shading-only and therefore cannot alter geometry, opacity,
    plane depth, ownership or the macroscopic normal consumed by water SSR.

    The optional wet response changes only linear-RGB Albedo brightness.  It
    models the rapid initial loss and later saturation of diffuse reflectance
    caused by pore water; it deliberately leaves Roughness, micro normals and
    dielectric specular unchanged.  A free-water film remains a separate P4
    optical layer, so applying water-like Fresnel here would double count it.
    """

    if base_albedo_srgb.ndim != 3 or base_albedo_srgb.shape[0] != 3:
        raise ValueError("base_albedo_srgb must have shape [3,H,W]")
    expected_scalar_shape = (1,) + tuple(base_albedo_srgb.shape[1:])
    if base_roughness.shape != expected_scalar_shape:
        raise ValueError("base_roughness must have shape [1,H,W]")
    if valid.shape != expected_scalar_shape:
        raise ValueError("valid must match base_roughness")
    if sampled["albedo_linear"].shape != base_albedo_srgb.shape:
        raise ValueError("sampled albedo must match base_albedo_srgb")
    if sampled["roughness"].shape != expected_scalar_shape:
        raise ValueError("sampled roughness must match base_roughness")
    if sampled["normal_tangent"].shape != base_albedo_srgb.shape:
        raise ValueError("sampled tangent normal must have shape [3,H,W]")
    if len(albedo_gain) != 3 or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in albedo_gain
    ):
        raise ValueError(
            "sand texture-primary albedo gain must be three finite "
            "non-negative values"
        )
    roughness_scale = float(roughness_scale)
    normal_strength = float(normal_strength)
    if not math.isfinite(roughness_scale) or roughness_scale <= 0.0:
        raise ValueError("sand texture-primary roughness scale must be positive")
    if not math.isfinite(normal_strength) or normal_strength < 0.0:
        raise ValueError("sand texture normal strength must be finite and non-negative")
    if not 0.0 <= roughness_min <= roughness_max <= 1.0:
        raise ValueError("sand roughness bounds must lie in [0,1]")
    wet_adjustment_enabled = bool(wet_adjustment_enabled)
    wet_albedo_scale = float(wet_albedo_scale)
    wet_response_rate = float(wet_response_rate)
    if not math.isfinite(wet_albedo_scale) or not 0.0 <= wet_albedo_scale <= 1.0:
        raise ValueError("sand wet Albedo scale must lie in [0,1]")
    if not math.isfinite(wet_response_rate) or wet_response_rate <= 0.0:
        raise ValueError("sand wet response rate must be finite and positive")
    if wetness is not None:
        if wetness.shape != expected_scalar_shape:
            raise ValueError("wetness must have shape [1,H,W]")
        if not torch.isfinite(wetness).all():
            raise ValueError("wetness must contain only finite values")
    if wet_adjustment_enabled and wetness is None:
        raise ValueError("wetness is required when wet adjustment is enabled")

    gain = torch.as_tensor(
        tuple(float(value) for value in albedo_gain),
        device=base_albedo_srgb.device,
        dtype=base_albedo_srgb.dtype,
    )[:, None, None]
    # Decode/encode boundaries remain explicit: the texture loader and mip
    # pyramid store Albedo in linear RGB, while the surrounding PGSR renderer
    # exchanges display/sRGB colors at this interface.
    primary_albedo_linear = torch.clamp(
        sampled["albedo_linear"] * gain, min=0.0
    )
    wet_response = torch.zeros(
        expected_scalar_shape,
        device=base_albedo_srgb.device,
        dtype=base_albedo_srgb.dtype,
    )
    wet_brightness = torch.ones_like(wet_response)
    if wet_adjustment_enabled:
        # Normalize the exponential so response(0)=0 and response(1)=1.
        # Multiplying every linear-RGB channel by the same factor reduces
        # diffuse brightness without inventing an unsupported wet-sand hue.
        bounded_wetness = torch.clamp(wetness, 0.0, 1.0).to(
            device=base_albedo_srgb.device,
            dtype=base_albedo_srgb.dtype,
        )
        denominator = -math.expm1(-wet_response_rate)
        wet_response = -torch.expm1(
            -wet_response_rate * bounded_wetness
        ) / denominator
        wet_brightness = 1.0 - (1.0 - wet_albedo_scale) * wet_response
        primary_albedo_linear = primary_albedo_linear * wet_brightness
    primary_albedo = _linear_to_srgb(primary_albedo_linear)
    primary_roughness = torch.clamp(
        sampled["roughness"] * roughness_scale,
        min=float(roughness_min),
        max=float(roughness_max),
    )

    tangent_normal = sampled["normal_tangent"].clone()
    tangent_normal[:2] *= normal_strength
    tangent_normal = F.normalize(tangent_normal, dim=0, eps=1.0e-8)
    macro_normal = F.normalize(macro_normal_world, dim=0, eps=1.0e-8)
    if "normal_world_override" in sampled:
        shading_normal = F.normalize(
            sampled["normal_world_override"], dim=0, eps=1.0e-8
        )
    else:
        tangent, bitangent = build_simulation_tangent_basis(
            macro_normal_world, sim_to_gs
        )
        shading_normal = F.normalize(
            tangent * tangent_normal[0:1]
            + bitangent * tangent_normal[1:2]
            + macro_normal * tangent_normal[2:3],
            dim=0,
            eps=1.0e-8,
        )

    valid_bool = valid.to(torch.bool)
    valid_rgb = valid_bool.repeat(3, 1, 1)
    return {
        "albedo_srgb": torch.where(
            valid_rgb, primary_albedo, base_albedo_srgb
        ),
        "roughness": torch.where(
            valid_bool, primary_roughness, base_roughness
        ),
        "normal_tangent": tangent_normal,
        "normal_world": torch.where(
            valid_rgb, shading_normal, macro_normal_world
        ),
        # Keep invalid/non-sand pixels neutral in diagnostics as well as in
        # material output; the response must never leak beyond texture-valid
        # sand ownership.
        "wet_response": torch.where(
            valid_bool, wet_response, torch.zeros_like(wet_response)
        ),
        "wet_brightness": torch.where(
            valid_bool, wet_brightness, torch.ones_like(wet_brightness)
        ),
    }


def sand_texture_uv_checker(uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return an eight-by-eight repeat checker for projection diagnostics."""

    cells = torch.floor((uv - torch.floor(uv)) * 8.0).to(torch.int64)
    checker = ((cells[0] + cells[1]) % 2).to(uv.dtype)[None]
    color_a = torch.tensor((0.10, 0.35, 0.85), device=uv.device, dtype=uv.dtype)
    color_b = torch.tensor((0.95, 0.70, 0.10), device=uv.device, dtype=uv.dtype)
    image = color_a[:, None, None] * (1.0 - checker) + color_b[:, None, None] * checker
    return image * valid.to(uv.dtype)
