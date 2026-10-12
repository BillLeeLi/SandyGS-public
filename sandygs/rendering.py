"""Small, device-independent helpers for the SandyGS material renderer.

This module deliberately does not import the CUDA rasterizer.  The equations
that turn rasterized P3 buffers into coverage, normals and view directions can
therefore be regression-tested on CPU before loading a multi-million Gaussian
scene.  Tensor layouts are documented at each boundary because P4 will consume
the same G-buffer contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_NVDIFFRAST_CONTEXTS: dict[str, object] = {}


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """Apply cubic smoothstep to a tensor already intended for ``[0,1]``."""

    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def align_reference_normals(
    reference_normal: torch.Tensor,
    shading_normal: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align an independent mesh normal with the rendered shading normal.

    Triangle winding is irrelevant to the T4 shading-band audit: the audit
    compares the *plane orientation* represented by the retained mesh with the
    alpha-blended Gaussian normal.  Flip the reference into the same hemisphere
    before measuring their angular disagreement, and zero pixels outside the
    shared valid support so saved diagnostic maps have an explicit background.

    Returns ``(aligned_reference, angle_degrees)`` with shapes ``[3,H,W]`` and
    ``[1,H,W]`` respectively.  This helper is diagnostic-only and never changes
    the production depth or shading normal.
    """

    if reference_normal.ndim != 3 or reference_normal.shape[0] != 3:
        raise ValueError("reference_normal must have shape [3,H,W]")
    if shading_normal.shape != reference_normal.shape:
        raise ValueError("shading_normal must match reference_normal")
    if valid.shape != (1,) + tuple(reference_normal.shape[1:]):
        raise ValueError("valid must have shape [1,H,W]")

    reference = F.normalize(reference_normal, dim=0, eps=1.0e-8)
    shading = F.normalize(shading_normal, dim=0, eps=1.0e-8)
    signed_dot = torch.sum(reference * shading, dim=0, keepdim=True)
    aligned = torch.where(
        (signed_dot < 0.0).repeat(3, 1, 1), -reference, reference
    )
    cosine = torch.clamp(
        torch.sum(aligned * shading, dim=0, keepdim=True), -1.0, 1.0
    )
    angle = torch.rad2deg(torch.acos(cosine))
    aligned = torch.where(valid.repeat(3, 1, 1), aligned, torch.zeros_like(aligned))
    angle = torch.where(valid, angle, torch.zeros_like(angle))
    return aligned, angle


def rasterize_triangle_reference(
    triangle_vertices_world: torch.Tensor,
    world_view_transform: torch.Tensor,
    full_projection_transform: torch.Tensor,
    image_height: int,
    image_width: int,
    triangle_vertex_sand_height: torch.Tensor | None = None,
    triangle_face_wetness: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Rasterize the final sand mesh independently of its Gaussian splats.

    The returned hard mask answers only which camera rays intersect a retained
    sand triangle.  It deliberately ignores Gaussian opacity, scale, plane
    reconstruction and source-scene ownership, so T2.1 can use it as a causal
    reference rather than another view of the production pipeline.
    """

    if triangle_vertices_world.ndim != 3 or triangle_vertices_world.shape[1:] != (3, 3):
        raise ValueError("triangle_vertices_world must have shape [T,3,3]")
    if triangle_vertex_sand_height is not None and (
        triangle_vertex_sand_height.shape
        != triangle_vertices_world.shape[:2]
    ):
        raise ValueError(
            "triangle_vertex_sand_height must have shape [T,3]"
        )
    if triangle_face_wetness is not None and (
        triangle_face_wetness.shape != triangle_vertices_world.shape[:1]
    ):
        raise ValueError("triangle_face_wetness must have shape [T]")
    if not triangle_vertices_world.is_cuda:
        raise ValueError("triangle reference rasterization requires CUDA")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("reference image dimensions must be positive")
    if triangle_vertices_world.shape[0] == 0:
        shape = (1, image_height, image_width)
        return {
            "sand_triangle_reference_coverage": torch.zeros(
                shape, dtype=torch.bool, device=triangle_vertices_world.device
            ),
            "sand_triangle_reference_depth_raw": torch.zeros(
                shape, dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
            "sand_triangle_reference_sand_height_raw": torch.zeros(
                shape, dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
            "sand_triangle_reference_face_wetness_raw": torch.zeros(
                shape, dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
            "sand_triangle_reference_normal_world_raw": torch.zeros(
                (3, image_height, image_width),
                dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
            "sand_triangle_reference_projected_max_edge_raw": torch.zeros(
                shape, dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
            "sand_triangle_reference_projected_area_raw": torch.zeros(
                shape, dtype=triangle_vertices_world.dtype,
                device=triangle_vertices_world.device,
            ),
        }

    # Import lazily so CPU-only unit tests retain the renderer-independent
    # contract of this module.
    import nvdiffrast.torch as dr

    device_key = str(triangle_vertices_world.device)
    context = _NVDIFFRAST_CONTEXTS.get(device_key)
    if context is None:
        context = dr.RasterizeCudaContext(device=triangle_vertices_world.device)
        _NVDIFFRAST_CONTEXTS[device_key] = context

    vertices = triangle_vertices_world.reshape(-1, 3)
    ones = torch.ones(
        (vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device
    )
    homogeneous = torch.cat((vertices, ones), dim=1)
    clip_vertices = homogeneous @ full_projection_transform
    triangle_indices = torch.arange(
        vertices.shape[0], dtype=torch.int32, device=vertices.device
    ).reshape(-1, 3)
    raster, _ = dr.rasterize(
        context,
        clip_vertices.unsqueeze(0),
        triangle_indices,
        resolution=[image_height, image_width],
    )

    camera_vertices = (
        vertices @ world_view_transform[:3, :3]
        + world_view_transform[3, :3]
    )
    triangle_world = triangle_vertices_world
    face_normal_world = F.normalize(
        torch.linalg.cross(
            triangle_world[:, 1] - triangle_world[:, 0],
            triangle_world[:, 2] - triangle_world[:, 0],
            dim=1,
        ),
        dim=1,
        eps=1.0e-8,
    )
    vertex_normal_world = (
        face_normal_world[:, None, :]
        .expand(-1, 3, -1)
        .reshape(-1, 3)
    )

    # T3's prerequisite gate needs to know whether the remaining geometric
    # disagreement is concentrated on genuinely large screen-space triangles.
    # Carry constant per-face edge/area attributes through this same reference
    # pass; no additional rasterization is required.
    clip_triangle = clip_vertices.reshape(-1, 3, 4)
    clip_w = clip_triangle[:, :, 3:4]
    safe_w = torch.where(
        torch.abs(clip_w) > 1.0e-8,
        clip_w,
        torch.ones_like(clip_w),
    )
    ndc_xy = clip_triangle[:, :, :2] / safe_w
    pixel_scale = torch.tensor(
        (0.5 * image_width, 0.5 * image_height),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    pixel_xy = ndc_xy * pixel_scale
    edge_lengths = torch.linalg.vector_norm(
        torch.stack(
            (
                pixel_xy[:, 1] - pixel_xy[:, 0],
                pixel_xy[:, 2] - pixel_xy[:, 1],
                pixel_xy[:, 0] - pixel_xy[:, 2],
            ),
            dim=1,
        ),
        dim=2,
    )
    projected_max_edge = torch.max(edge_lengths, dim=1).values
    projected_area = 0.5 * torch.abs(
        (pixel_xy[:, 1, 0] - pixel_xy[:, 0, 0])
        * (pixel_xy[:, 2, 1] - pixel_xy[:, 0, 1])
        - (pixel_xy[:, 1, 1] - pixel_xy[:, 0, 1])
        * (pixel_xy[:, 2, 0] - pixel_xy[:, 0, 0])
    )
    projected_max_edge = torch.nan_to_num(
        projected_max_edge, nan=0.0, posinf=0.0, neginf=0.0
    )
    projected_area = torch.nan_to_num(
        projected_area, nan=0.0, posinf=0.0, neginf=0.0
    )
    vertex_projected_max_edge = (
        projected_max_edge[:, None, None]
        .expand(-1, 3, 1)
        .reshape(-1, 1)
    )
    vertex_projected_area = (
        projected_area[:, None, None]
        .expand(-1, 3, 1)
        .reshape(-1, 1)
    )
    if triangle_vertex_sand_height is None:
        vertex_sand_height = torch.zeros(
            (vertices.shape[0], 1), dtype=vertices.dtype,
            device=vertices.device,
        )
    else:
        vertex_sand_height = triangle_vertex_sand_height.reshape(
            -1, 1
        ).to(device=vertices.device, dtype=vertices.dtype)
    if triangle_face_wetness is None:
        vertex_face_wetness = torch.zeros(
            (vertices.shape[0], 1), dtype=vertices.dtype,
            device=vertices.device,
        )
    else:
        # The triangle_games backend creates exactly one Gaussian per final
        # triangle. Repeat that primitive's pre-rasterization wetness at all
        # three vertices so nvdiffrast produces an independent per-face field
        # without blending neighboring Gaussian footprints.
        vertex_face_wetness = (
            triangle_face_wetness[:, None, None]
            .expand(-1, 3, 1)
            .reshape(-1, 1)
            .to(device=vertices.device, dtype=vertices.dtype)
        )
    interpolated_attributes, _ = dr.interpolate(
        torch.cat(
            (
                camera_vertices[:, 2:3],
                vertex_sand_height,
                vertex_face_wetness,
                vertex_normal_world,
                vertex_projected_max_edge,
                vertex_projected_area,
            ),
            dim=1,
        ).unsqueeze(0).contiguous(),
        raster.contiguous(),
        triangle_indices.contiguous(),
    )
    # SandyGS' full projection matrix already incorporates its image-space Y
    # convention. Applying an additional OpenGL-to-image flip here would mirror
    # the reference relative to the plane Gaussian rasterizer.
    coverage = (raster[0, :, :, 3] > 0.0).unsqueeze(0)
    depth = interpolated_attributes[0, :, :, 0].unsqueeze(0)
    sand_height = interpolated_attributes[0, :, :, 1].unsqueeze(0)
    face_wetness = interpolated_attributes[0, :, :, 2].unsqueeze(0)
    normal_world = interpolated_attributes[0, :, :, 3:6].permute(2, 0, 1)
    projected_max_edge_map = interpolated_attributes[
        0, :, :, 6
    ].unsqueeze(0)
    projected_area_map = interpolated_attributes[0, :, :, 7].unsqueeze(0)
    depth = torch.where(coverage, depth, torch.zeros_like(depth))
    sand_height = torch.where(
        coverage, sand_height, torch.zeros_like(sand_height)
    )
    face_wetness = torch.where(
        coverage, face_wetness, torch.zeros_like(face_wetness)
    )
    normal_world = torch.where(
        coverage.repeat(3, 1, 1),
        F.normalize(normal_world, dim=0, eps=1.0e-8),
        torch.zeros_like(normal_world),
    )
    projected_max_edge_map = torch.where(
        coverage,
        projected_max_edge_map,
        torch.zeros_like(projected_max_edge_map),
    )
    projected_area_map = torch.where(
        coverage,
        projected_area_map,
        torch.zeros_like(projected_area_map),
    )
    return {
        "sand_triangle_reference_coverage": coverage,
        "sand_triangle_reference_depth_raw": depth,
        "sand_triangle_reference_sand_height_raw": sand_height,
        "sand_triangle_reference_face_wetness_raw": face_wetness,
        "sand_triangle_reference_normal_world_raw": normal_world,
        "sand_triangle_reference_projected_max_edge_raw": (
            projected_max_edge_map
        ),
        "sand_triangle_reference_projected_area_raw": projected_area_map,
    }


def sand_height_to_soft_coverage(
    sand_height: torch.Tensor,
    reference_coverage: torch.Tensor,
    *,
    sand_epsilon: float,
    fade_height: float,
) -> torch.Tensor:
    """Turn perspective-interpolated physical sand thickness into coverage.

    Unlike Gaussian alpha, this target is defined entirely by the simulated
    height field.  The hard triangle mask only suppresses interpolation
    background; clipped boundary vertices already carry ``sand_epsilon``, so
    the ramp approaches zero continuously at the physical material boundary.
    """

    if sand_height.shape != reference_coverage.shape:
        raise ValueError("sand height and reference coverage shapes must match")
    sand_epsilon = _finite_scalar(sand_epsilon, "sand_epsilon")
    fade_height = _finite_scalar(fade_height, "fade_height")
    if sand_epsilon < 0.0 or fade_height <= sand_epsilon:
        raise ValueError("fade_height must exceed non-negative sand_epsilon")
    normalized = (sand_height - sand_epsilon) / (
        fade_height - sand_epsilon
    )
    return smoothstep01(normalized) * reference_coverage.to(
        dtype=sand_height.dtype
    )


def compare_sand_soft_boundary(
    target_coverage: torch.Tensor,
    gaussian_coverage: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare physical soft coverage with the current Gaussian silhouette.

    Positive signed error means the Gaussian extends beyond/exceeds the target;
    negative error means the physical height field is under-represented.  RGB
    diagnostics use red=over, blue=under and green=matched coverage.
    """

    if target_coverage.shape != gaussian_coverage.shape:
        raise ValueError("soft-boundary coverage inputs must have equal shapes")
    if target_coverage.ndim != 3 or target_coverage.shape[0] != 1:
        raise ValueError("soft-boundary coverage inputs must have shape [1,H,W]")
    target = torch.clamp(target_coverage, 0.0, 1.0)
    current = torch.clamp(gaussian_coverage, 0.0, 1.0)
    signed_error = current - target
    under = torch.clamp(-signed_error, 0.0, 1.0)
    over = torch.clamp(signed_error, 0.0, 1.0)
    matched = torch.minimum(target, current)
    error_visualization = torch.cat((over, matched, under), dim=0)
    # Red-only means current Gaussian coverage, green-only means the physical
    # target, and yellow means agreement in this direct silhouette overlay.
    target_current_overlay = torch.cat(
        (current, target, torch.zeros_like(target)), dim=0
    )
    return {
        "sand_target_soft_coverage": target,
        "sand_boundary_current_coverage_raw": current,
        "sand_boundary_under_coverage": under,
        "sand_boundary_over_coverage": over,
        "sand_boundary_coverage_match": matched,
        "sand_boundary_signed_error_raw": signed_error,
        "sand_boundary_error_visualization": error_visualization,
        "sand_boundary_target_current_overlay": target_current_overlay,
    }


def guide_sand_coverage_with_soft_boundary(
    gaussian_coverage: torch.Tensor,
    target_coverage: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Conservatively clip production sand coverage to the physical target.

    T2.2 showed that the current Gaussian support already covers essentially
    the entire physical soft target. T2.4 therefore never invents material
    attributes where the splats have no support: it can only retain or reduce
    current coverage. Keeping the unguided value in the return contract makes
    the production change directly auditable in the same render.
    """

    if gaussian_coverage.shape != target_coverage.shape:
        raise ValueError("guided sand coverage inputs must have equal shapes")
    if gaussian_coverage.ndim != 3 or gaussian_coverage.shape[0] != 1:
        raise ValueError("guided sand coverage inputs must have shape [1,H,W]")
    current = torch.clamp(gaussian_coverage, 0.0, 1.0)
    target = torch.clamp(target_coverage, 0.0, 1.0)
    guided = torch.minimum(current, target) if enabled else current
    return {
        "sand_coverage_unguided": current,
        "sand_guided_coverage": guided,
        "sand_guidance_removed_coverage": torch.clamp(
            current - guided, min=0.0
        ),
        "sand_guidance_residual_undercoverage": torch.clamp(
            target - guided, min=0.0
        ),
    }


def classify_sand_t3_prerequisites(
    target_coverage: torch.Tensor,
    guided_coverage: torch.Tensor,
    sand_owner: torch.Tensor,
    resolved_depth: torch.Tensor,
    center_depth: torch.Tensor,
    center_depth_std: torch.Tensor,
    gaussian_normal_world: torch.Tensor,
    reference_coverage: torch.Tensor,
    reference_depth: torch.Tensor,
    reference_normal_world: torch.Tensor,
    reference_projected_max_edge: torch.Tensor,
    reference_projected_area: torch.Tensor,
    *,
    core_coverage_min: float = 0.95,
    erode_kernel: int = 5,
    depth_absolute_tolerance: float = 0.005,
    depth_relative_tolerance: float = 0.002,
    normal_angle_tolerance_degrees: float = 15.0,
    center_std_relative_tolerance: float = 0.01,
    large_projected_edge_pixels: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Audit whether T2.4 still contains errors that finer triangles can fix.

    The audit deliberately excludes the antialiased physical boundary and any
    sand hidden by the source scene. T2.4 already owns those decisions. T3 is
    justified only by residual depth/normal mixing inside a dense, visible
    physical sand surface, especially when it correlates with large projected
    triangles.
    """

    scalar_inputs = (
        target_coverage,
        guided_coverage,
        sand_owner,
        resolved_depth,
        center_depth,
        center_depth_std,
        reference_coverage,
        reference_depth,
        reference_projected_max_edge,
        reference_projected_area,
    )
    shape = target_coverage.shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError("T3 prerequisite scalar maps must have shape [1,H,W]")
    if any(value.shape != shape for value in scalar_inputs):
        raise ValueError("all T3 prerequisite scalar maps must have equal shapes")
    normal_shape = (3, *shape[-2:])
    if (
        gaussian_normal_world.shape != normal_shape
        or reference_normal_world.shape != normal_shape
    ):
        raise ValueError("T3 prerequisite normal maps must have shape [3,H,W]")

    core_coverage_min = _finite_scalar(
        core_coverage_min, "core_coverage_min"
    )
    depth_absolute_tolerance = _finite_scalar(
        depth_absolute_tolerance, "depth_absolute_tolerance"
    )
    depth_relative_tolerance = _finite_scalar(
        depth_relative_tolerance, "depth_relative_tolerance"
    )
    normal_angle_tolerance_degrees = _finite_scalar(
        normal_angle_tolerance_degrees,
        "normal_angle_tolerance_degrees",
    )
    center_std_relative_tolerance = _finite_scalar(
        center_std_relative_tolerance,
        "center_std_relative_tolerance",
    )
    large_projected_edge_pixels = _finite_scalar(
        large_projected_edge_pixels, "large_projected_edge_pixels"
    )
    if not 0.0 <= core_coverage_min <= 1.0:
        raise ValueError("core_coverage_min must lie in [0,1]")
    if erode_kernel < 1 or erode_kernel % 2 == 0:
        raise ValueError("erode_kernel must be a positive odd integer")
    if (
        depth_absolute_tolerance < 0.0
        or depth_relative_tolerance < 0.0
        or not 0.0 <= normal_angle_tolerance_degrees <= 180.0
        or center_std_relative_tolerance < 0.0
        or large_projected_edge_pixels < 0.0
    ):
        raise ValueError("T3 prerequisite tolerances must be non-negative")

    physical_core = target_coverage >= core_coverage_min
    radius = erode_kernel // 2
    padded_holes = F.pad(
        (~physical_core).to(target_coverage.dtype)[None, ...],
        (radius,) * 4,
        mode="constant",
        value=1.0,
    )
    physical_interior = physical_core & (
        F.max_pool2d(
            padded_holes,
            kernel_size=erode_kernel,
            stride=1,
            padding=0,
        )[0]
        <= 0.0
    )
    numeric_depth = (
        torch.isfinite(resolved_depth)
        & (resolved_depth > 0.0)
        & torch.isfinite(reference_depth)
        & (reference_depth > 0.0)
    )
    numeric_center = (
        torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_depth_std)
        & (center_depth_std >= 0.0)
    )
    numeric_normal = (
        torch.isfinite(gaussian_normal_world).all(dim=0, keepdim=True)
        & torch.isfinite(reference_normal_world).all(dim=0, keepdim=True)
    )
    audit = (
        physical_interior
        & (guided_coverage >= 0.5)
        & sand_owner.to(torch.bool)
        & reference_coverage.to(torch.bool)
        & numeric_depth
        & numeric_center
        & numeric_normal
    )

    depth_residual = torch.abs(resolved_depth - reference_depth)
    depth_tolerance = torch.maximum(
        torch.full_like(reference_depth, depth_absolute_tolerance),
        depth_relative_tolerance * reference_depth,
    )
    depth_inconsistent = audit & (depth_residual > depth_tolerance)

    gaussian_normal = F.normalize(gaussian_normal_world, dim=0, eps=1.0e-8)
    reference_normal = F.normalize(reference_normal_world, dim=0, eps=1.0e-8)
    # Plane orientation does not change the represented tangent surface. Use
    # |dot| so an otherwise identical sign convention is not counted as a T3
    # error; shading orientation is audited separately by the G-buffer checks.
    normal_cosine = torch.clamp(
        torch.abs((gaussian_normal * reference_normal).sum(0, keepdim=True)),
        0.0,
        1.0,
    )
    normal_angle_degrees = torch.rad2deg(torch.acos(normal_cosine))
    normal_mismatch = audit & (
        normal_angle_degrees > normal_angle_tolerance_degrees
    )

    center_relative_std = center_depth_std / torch.clamp(
        torch.abs(center_depth), min=1.0e-6
    )
    center_mixed = audit & (
        center_relative_std > center_std_relative_tolerance
    )
    large_triangle = audit & (
        reference_projected_max_edge >= large_projected_edge_pixels
    )
    any_failure = depth_inconsistent | normal_mismatch | center_mixed
    multiple_failure = (
        depth_inconsistent.to(torch.int32)
        + normal_mismatch.to(torch.int32)
        + center_mixed.to(torch.int32)
    ) >= 2

    classification = torch.zeros(
        normal_shape,
        dtype=target_coverage.dtype,
        device=target_coverage.device,
    )
    classification[:, audit[0]] = torch.tensor(
        (0.22, 0.22, 0.22),
        dtype=classification.dtype,
        device=classification.device,
    )[:, None]
    depth_only = depth_inconsistent & ~normal_mismatch & ~center_mixed
    normal_only = normal_mismatch & ~depth_inconsistent & ~center_mixed
    center_only = center_mixed & ~depth_inconsistent & ~normal_mismatch
    classification[:, depth_only[0]] = torch.tensor(
        (1.0, 0.0, 0.0),
        dtype=classification.dtype,
        device=classification.device,
    )[:, None]
    classification[:, normal_only[0]] = torch.tensor(
        (0.0, 1.0, 0.0),
        dtype=classification.dtype,
        device=classification.device,
    )[:, None]
    classification[:, center_only[0]] = torch.tensor(
        (0.0, 0.35, 1.0),
        dtype=classification.dtype,
        device=classification.device,
    )[:, None]
    classification[:, multiple_failure[0]] = torch.tensor(
        (1.0, 0.0, 1.0),
        dtype=classification.dtype,
        device=classification.device,
    )[:, None]

    return {
        "sand_t3_physical_interior": physical_interior,
        "sand_t3_audit_region": audit,
        "sand_t3_depth_inconsistent": depth_inconsistent,
        "sand_t3_normal_mismatch": normal_mismatch,
        "sand_t3_center_depth_mixed": center_mixed,
        "sand_t3_any_failure": any_failure,
        "sand_t3_multiple_failure": multiple_failure,
        "sand_t3_large_triangle_support": large_triangle,
        "sand_t3_failure_on_large_triangle": any_failure & large_triangle,
        "sand_t3_failure_classification": classification,
        "sand_t3_depth_residual_raw": torch.where(
            audit, depth_residual, torch.zeros_like(depth_residual)
        ),
        "sand_t3_depth_tolerance_raw": torch.where(
            audit, depth_tolerance, torch.zeros_like(depth_tolerance)
        ),
        "sand_t3_normal_angle_degrees_raw": torch.where(
            audit, normal_angle_degrees, torch.zeros_like(normal_angle_degrees)
        ),
        "sand_t3_center_depth_relative_std_raw": torch.where(
            audit, center_relative_std, torch.zeros_like(center_relative_std)
        ),
        "sand_t3_projected_max_edge_raw": torch.where(
            audit,
            reference_projected_max_edge,
            torch.zeros_like(reference_projected_max_edge),
        ),
        "sand_t3_projected_area_raw": torch.where(
            audit,
            reference_projected_area,
            torch.zeros_like(reference_projected_area),
        ),
    }


def classify_sand_primitive_overcoverage(
    target_coverage: torch.Tensor,
    gaussian_coverage: torch.Tensor,
    hard_reference_coverage: torch.Tensor,
    steep_front_alpha: torch.Tensor,
    steep_rear_alpha: torch.Tensor,
    other_alpha: torch.Tensor,
    resolved_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    *,
    overcoverage_epsilon: float = 1.0e-6,
    support_epsilon: float = 1.0e-8,
    depth_absolute_tolerance: float = 0.005,
    depth_relative_tolerance: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Attribute T2.2 overcoverage to disjoint primitive subsets.

    The three subset alpha images are rendered independently, so their values
    are not additive when splats overlap. We therefore use only the dominant
    subset at each overcovered pixel and explicitly label the result as an
    attribution diagnostic, not an exact alpha decomposition.

    Mesh-depth consistency is evaluated only where the hard triangle pass has
    a valid nearest surface. Strict-outside pixels are reported separately;
    assigning them a fictitious mesh depth would hide the very contour leak
    that T2.3 is intended to diagnose.
    """

    tensors = (
        gaussian_coverage,
        hard_reference_coverage,
        steep_front_alpha,
        steep_rear_alpha,
        other_alpha,
        resolved_depth,
        reference_depth,
    )
    if target_coverage.ndim != 3 or target_coverage.shape[0] != 1:
        raise ValueError("primitive diagnostic buffers must have shape [1,H,W]")
    if any(tensor.shape != target_coverage.shape for tensor in tensors):
        raise ValueError("primitive diagnostic buffer shapes must match")
    overcoverage_epsilon = _finite_scalar(
        overcoverage_epsilon, "overcoverage_epsilon"
    )
    support_epsilon = _finite_scalar(support_epsilon, "support_epsilon")
    depth_absolute_tolerance = _finite_scalar(
        depth_absolute_tolerance, "depth_absolute_tolerance"
    )
    depth_relative_tolerance = _finite_scalar(
        depth_relative_tolerance, "depth_relative_tolerance"
    )
    if min(
        overcoverage_epsilon,
        support_epsilon,
        depth_absolute_tolerance,
        depth_relative_tolerance,
    ) < 0.0:
        raise ValueError("primitive diagnostic tolerances must be non-negative")

    target = torch.clamp(target_coverage, 0.0, 1.0)
    current = torch.clamp(gaussian_coverage, 0.0, 1.0)
    over = torch.clamp(current - target, min=0.0)
    over_mask = over > overcoverage_epsilon
    hard_reference = hard_reference_coverage.to(torch.bool)

    # Class order is rear/front/other so the RGB visualization can use the
    # same semantic order: red=suspected rear wall, green=front, blue=other.
    subset_alpha = torch.cat(
        (
            torch.clamp(steep_rear_alpha, 0.0, 1.0),
            torch.clamp(steep_front_alpha, 0.0, 1.0),
            torch.clamp(other_alpha, 0.0, 1.0),
        ),
        dim=0,
    )
    dominant_alpha, dominant_index = torch.max(subset_alpha, dim=0, keepdim=True)
    attributed = over_mask & (dominant_alpha > support_epsilon)
    rear = attributed & (dominant_index == 0)
    front = attributed & (dominant_index == 1)
    other = attributed & (dominant_index == 2)
    unattributed = over_mask & ~attributed
    primitive_classification = torch.cat(
        (
            rear.to(over.dtype) + unattributed.to(over.dtype),
            front.to(over.dtype) + unattributed.to(over.dtype),
            other.to(over.dtype),
        ),
        dim=0,
    )

    reference_valid = (
        hard_reference
        & torch.isfinite(reference_depth)
        & (reference_depth > 0.0)
    )
    resolved_valid = torch.isfinite(resolved_depth) & (resolved_depth > 0.0)
    depth_evaluated = over_mask & reference_valid & resolved_valid
    depth_tolerance = torch.maximum(
        torch.full_like(reference_depth, depth_absolute_tolerance),
        torch.abs(reference_depth) * depth_relative_tolerance,
    )
    depth_residual = resolved_depth - reference_depth
    depth_inconsistent = depth_evaluated & (
        torch.abs(depth_residual) > depth_tolerance
    )
    depth_consistent = depth_evaluated & ~depth_inconsistent
    depth_no_reference = over_mask & ~reference_valid
    depth_invalid = over_mask & reference_valid & ~resolved_valid
    # Depth colors: red=inconsistent, green=consistent, blue=no hard-mesh
    # reference, yellow=invalid resolved depth despite a valid mesh reference.
    depth_classification = torch.cat(
        (
            depth_inconsistent.to(over.dtype) + depth_invalid.to(over.dtype),
            depth_consistent.to(over.dtype) + depth_invalid.to(over.dtype),
            depth_no_reference.to(over.dtype),
        ),
        dim=0,
    )

    return {
        "sand_steep_rear_overcoverage": over * rear.to(over.dtype),
        "sand_steep_front_overcoverage": over * front.to(over.dtype),
        "sand_other_primitive_overcoverage": over * other.to(over.dtype),
        "sand_unattributed_overcoverage": over * unattributed.to(over.dtype),
        "sand_overcoverage_strict_outside": (
            over * (~hard_reference).to(over.dtype)
        ),
        "sand_overcoverage_inside_hard_mesh": (
            over * hard_reference.to(over.dtype)
        ),
        "sand_overcoverage_primitive_classification": primitive_classification,
        "sand_mesh_depth_inconsistent_support": (
            over * depth_inconsistent.to(over.dtype)
        ),
        "sand_mesh_depth_consistency_classification": depth_classification,
        "sand_mesh_depth_residual_raw": torch.where(
            depth_evaluated, depth_residual, torch.zeros_like(depth_residual)
        ),
        "sand_mesh_depth_tolerance_raw": torch.where(
            depth_evaluated, depth_tolerance, torch.zeros_like(depth_tolerance)
        ),
        "sand_primitive_overcoverage_mask": over_mask,
        "sand_primitive_attribution_rear": rear,
        "sand_primitive_attribution_front": front,
        "sand_primitive_attribution_other": other,
        "sand_primitive_attribution_unattributed": unattributed,
        "sand_mesh_depth_evaluated": depth_evaluated,
        "sand_mesh_depth_inconsistent": depth_inconsistent,
        "sand_mesh_depth_consistent": depth_consistent,
        "sand_mesh_depth_no_reference": depth_no_reference,
        "sand_mesh_depth_invalid": depth_invalid,
    }


def classify_sand_triangle_footprint(
    reference_coverage: torch.Tensor,
    reference_depth: torch.Tensor,
    accumulated_alpha: torch.Tensor,
    coverage: torch.Tensor,
    resolved_depth: torch.Tensor,
    resolved_depth_valid: torch.Tensor,
    depth_in_front: torch.Tensor,
    final_owner: torch.Tensor,
    *,
    alpha_min: float,
    owner_coverage_min: float = 0.5,
    footprint_epsilon: float = 1.0e-8,
    boundary_kernel: int = 5,
) -> dict[str, torch.Tensor]:
    """Classify mesh-interior holes and Gaussian footprint overreach.

    Eroding/dilating the hard mesh mask reserves a narrow neutral boundary
    band.  Pixel-center rules and antialiasing can legitimately disagree there;
    only the eroded interior is called a missing mesh pixel, and only pixels
    outside the dilated mask are called Gaussian overcoverage.
    """

    tensors = (
        reference_depth, accumulated_alpha, coverage, resolved_depth,
        resolved_depth_valid, depth_in_front, final_owner,
    )
    if reference_coverage.ndim != 3 or reference_coverage.shape[0] != 1:
        raise ValueError("triangle diagnostic buffers must have shape [1,H,W]")
    if any(tensor.shape != reference_coverage.shape for tensor in tensors):
        raise ValueError("all triangle diagnostic buffers must have equal shapes")
    alpha_min = _finite_scalar(alpha_min, "alpha_min")
    owner_coverage_min = _finite_scalar(owner_coverage_min, "owner_coverage_min")
    footprint_epsilon = _finite_scalar(footprint_epsilon, "footprint_epsilon")
    if not 0.0 <= alpha_min <= 1.0 or not 0.0 <= owner_coverage_min <= 1.0:
        raise ValueError("diagnostic thresholds must lie in [0,1]")
    if footprint_epsilon < 0.0:
        raise ValueError("footprint_epsilon must be non-negative")
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be a positive odd integer")

    reference = reference_coverage.to(torch.bool)
    pooled_reference = F.max_pool2d(
        reference.to(torch.float32).unsqueeze(0),
        kernel_size=boundary_kernel,
        stride=1,
        padding=boundary_kernel // 2,
    ).squeeze(0).to(torch.bool)
    eroded_reference = ~F.max_pool2d(
        (~reference).to(torch.float32).unsqueeze(0),
        kernel_size=boundary_kernel,
        stride=1,
        padding=boundary_kernel // 2,
    ).squeeze(0).to(torch.bool)
    interior = reference & eroded_reference
    boundary = pooled_reference & ~interior
    strict_outside = ~pooled_reference

    has_footprint = accumulated_alpha > footprint_epsilon
    alpha_valid = accumulated_alpha > alpha_min
    coverage_owner = coverage >= owner_coverage_min
    depth_valid = resolved_depth_valid.to(torch.bool)
    in_front = depth_in_front >= 0.5
    owner = final_owner.to(torch.bool)

    def partition(region: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        no_footprint = region & ~has_footprint
        low_alpha = region & has_footprint & ~alpha_valid
        low_coverage = region & alpha_valid & ~coverage_owner
        invalid_depth = region & alpha_valid & coverage_owner & ~depth_valid
        behind_source = region & alpha_valid & coverage_owner & depth_valid & ~in_front
        valid_owner = region & owner
        claimed = (
            no_footprint | low_alpha | low_coverage | invalid_depth
            | behind_source | valid_owner
        )
        unclassified = region & ~claimed
        return {
            f"sand_triangle_{prefix}_no_footprint": no_footprint,
            f"sand_triangle_{prefix}_low_alpha": low_alpha,
            f"sand_triangle_{prefix}_low_coverage": low_coverage,
            f"sand_triangle_{prefix}_invalid_depth": invalid_depth,
            f"sand_triangle_{prefix}_behind_source": behind_source,
            f"sand_triangle_{prefix}_owner": valid_owner,
            f"sand_triangle_{prefix}_unclassified": unclassified,
        }

    internal = partition(interior, "internal")
    outside_support = strict_outside & has_footprint
    overcoverage = partition(outside_support, "overcoverage")

    def classification(
        groups: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        image = torch.zeros(
            (3, *reference.shape[1:]), dtype=coverage.dtype,
            device=coverage.device,
        )
        colors = {
            "no_footprint": (0.0, 0.35, 1.0),
            "low_alpha": (0.0, 1.0, 1.0),
            "low_coverage": (1.0, 1.0, 0.0),
            "invalid_depth": (1.0, 0.0, 1.0),
            "behind_source": (1.0, 0.5, 0.0),
            "owner": (0.0, 1.0, 0.0),
            "unclassified": (1.0, 0.0, 0.0),
        }
        for name, color in colors.items():
            mask = groups[f"sand_triangle_{prefix}_{name}"]
            color_tensor = torch.tensor(
                color, dtype=image.dtype, device=image.device
            ).view(3, 1, 1)
            image = torch.where(mask.repeat(3, 1, 1), color_tensor, image)
        return image

    overlap = interior & depth_valid & torch.isfinite(reference_depth)
    depth_residual = torch.zeros_like(resolved_depth)
    depth_residual[overlap] = resolved_depth[overlap] - reference_depth[overlap]
    return {
        "sand_triangle_reference_interior": interior,
        "sand_triangle_reference_boundary": boundary,
        "sand_triangle_reference_strict_outside": strict_outside,
        "sand_triangle_reference_depth_overlap": overlap,
        "sand_triangle_reference_depth_residual_raw": depth_residual,
        "sand_triangle_internal_failure_classification": classification(
            internal, "internal"
        ),
        "sand_triangle_overcoverage_classification": classification(
            overcoverage, "overcoverage"
        ),
        **internal,
        **overcoverage,
    }


def opaque_surface_coverage(
    accumulated_alpha: torch.Tensor,
    alpha_low: float,
    alpha_high: float,
) -> torch.Tensor:
    """Convert Gaussian alpha into opaque-surface pixel coverage.

    Alpha is *not* used as material transparency.  Values above ``alpha_high``
    mean that the pixel footprint is fully covered by an opaque aggregate;
    only the Gaussian tail between the two thresholds is retained as silhouette
    antialiasing.  Keeping this operation separate prevents P4 from mistaking
    antialiasing coverage for water/sand transmittance.
    """

    alpha_low = _finite_scalar(alpha_low, "alpha_low")
    alpha_high = _finite_scalar(alpha_high, "alpha_high")
    if not 0.0 <= alpha_low < alpha_high <= 1.0:
        raise ValueError(
            "opaque coverage thresholds must satisfy 0 <= low < high <= 1"
        )
    normalized = (accumulated_alpha - alpha_low) / (alpha_high - alpha_low)
    return smoothstep01(normalized)


def valid_visible_coverage(
    coverage: torch.Tensor,
    surface_valid: torch.Tensor,
    in_front: torch.Tensor,
) -> torch.Tensor:
    """Keep continuous RGB coverage only where surface geometry is valid.

    A Gaussian alpha footprint can remain high even when the plane-depth
    reconstruction is zero, negative or non-finite.  Such a pixel has no
    meaningful normal, Fresnel term or SSR ray and must reveal the already
    composed opaque substrate.  Multiplying validity here preserves soft
    antialiasing on valid silhouettes without letting an invalid surface
    replace the substrate with an uninitialized optical buffer.
    """

    if coverage.shape != surface_valid.shape or coverage.shape != in_front.shape:
        raise ValueError(
            "coverage, surface_valid and in_front must have equal shapes"
        )
    return (
        coverage
        * surface_valid.to(dtype=coverage.dtype)
        * in_front.to(dtype=coverage.dtype)
    )


def classify_water_gbuffer_stages(
    accumulated_alpha: torch.Tensor,
    coverage: torch.Tensor,
    plane_depth_valid: torch.Tensor,
    depth_fallback_used: torch.Tensor,
    depth_in_front: torch.Tensor,
    *,
    alpha_min: float,
    owner_coverage_min: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Partition every pixel by the first water G-buffer stage it reaches.

    This is a diagnostic-only view of the existing P4 gates.  It deliberately
    does not alter coverage, depth, occlusion, or final RGB.  The masks are
    mutually exclusive and exhaustive, making an apparent substrate fallback
    attributable to one precise stage instead of overloading the older
    ``water_invalid_coverage`` map.

    Classification colors are: black=no footprint, blue=alpha below the
    renderer threshold, magenta=invalid tangent-plane depth, orange=water
    behind the opaque source+sand layer, cyan=valid antialias-only footprint,
    and green=valid water-surface owner.
    """

    tensors = (
        accumulated_alpha,
        coverage,
        plane_depth_valid,
        depth_fallback_used,
        depth_in_front,
    )
    if any(tensor.shape != accumulated_alpha.shape for tensor in tensors[1:]):
        raise ValueError("all water G-buffer classification inputs must match")
    alpha_min = _finite_scalar(alpha_min, "alpha_min")
    owner_coverage_min = _finite_scalar(
        owner_coverage_min, "owner_coverage_min"
    )
    if not 0.0 <= alpha_min <= 1.0:
        raise ValueError("alpha_min must lie in [0,1]")
    if not 0.0 <= owner_coverage_min <= 1.0:
        raise ValueError("owner_coverage_min must lie in [0,1]")

    has_footprint = accumulated_alpha > 0.0
    alpha_valid = accumulated_alpha > alpha_min
    resolved_depth_valid = plane_depth_valid | depth_fallback_used
    no_footprint = ~has_footprint
    low_alpha = has_footprint & (~alpha_valid)
    invalid_plane_depth = alpha_valid & (~resolved_depth_valid)
    behind_opaque = alpha_valid & resolved_depth_valid & (~depth_in_front)
    valid_candidate = alpha_valid & resolved_depth_valid & depth_in_front
    valid_partial = valid_candidate & (coverage < owner_coverage_min)
    valid_owner = valid_candidate & (coverage >= owner_coverage_min)

    # Build the legend-colored map from disjoint masks. Black is already the
    # background/no-footprint class, so only the other five need assignments.
    classification_rgb = torch.zeros(
        (3, *accumulated_alpha.shape[-2:]),
        device=accumulated_alpha.device,
        dtype=accumulated_alpha.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(
            torch.where(mask.repeat(3, 1, 1), color_tensor, classification_rgb)
        )

    paint(low_alpha, (0.0, 0.35, 1.0))
    paint(invalid_plane_depth, (1.0, 0.0, 1.0))
    paint(behind_opaque, (1.0, 0.5, 0.0))
    paint(valid_partial, (0.0, 1.0, 1.0))
    paint(valid_owner, (0.0, 1.0, 0.0))
    return {
        "water_failure_no_footprint": no_footprint,
        "water_failure_low_alpha": low_alpha,
        "water_failure_invalid_plane_depth": invalid_plane_depth,
        "water_failure_behind_opaque": behind_opaque,
        "water_failure_valid_partial": valid_partial,
        "water_failure_valid_owner": valid_owner,
        "water_failure_classification": classification_rgb,
    }


def classify_sand_gbuffer_stages(
    accumulated_alpha: torch.Tensor,
    coverage: torch.Tensor,
    plane_depth_valid: torch.Tensor,
    depth_in_front: torch.Tensor,
    *,
    alpha_min: float,
    owner_coverage_min: float = 0.5,
    owner_in_front_min: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Explain the exact stage at which each sand pixel stops being an owner.

    Alpha and coverage describe only the projected Gaussian footprint.  Sand
    can still disappear from the composed RGB when its tangent-plane depth is
    invalid or when the source scene wins the soft depth test.  This diagnostic
    partition mirrors those production gates without changing any of them.

    Colors match the water diagnostic where possible: black=no footprint,
    blue=alpha below the sand threshold, magenta=invalid plane depth,
    orange=behind source, cyan=valid antialias-only coverage, green=owner.
    """

    tensors = (
        accumulated_alpha,
        coverage,
        plane_depth_valid,
        depth_in_front,
    )
    if any(tensor.shape != accumulated_alpha.shape for tensor in tensors[1:]):
        raise ValueError("all sand G-buffer classification inputs must match")
    alpha_min = _finite_scalar(alpha_min, "alpha_min")
    owner_coverage_min = _finite_scalar(
        owner_coverage_min, "owner_coverage_min"
    )
    owner_in_front_min = _finite_scalar(
        owner_in_front_min, "owner_in_front_min"
    )
    if not 0.0 <= alpha_min <= 1.0:
        raise ValueError("alpha_min must lie in [0,1]")
    if not 0.0 <= owner_coverage_min <= 1.0:
        raise ValueError("owner_coverage_min must lie in [0,1]")
    if not 0.0 <= owner_in_front_min <= 1.0:
        raise ValueError("owner_in_front_min must lie in [0,1]")

    has_footprint = accumulated_alpha > 0.0
    alpha_valid = accumulated_alpha > alpha_min
    in_front = depth_in_front >= owner_in_front_min
    no_footprint = ~has_footprint
    low_alpha = has_footprint & (~alpha_valid)
    invalid_plane_depth = alpha_valid & (~plane_depth_valid)
    behind_source = alpha_valid & plane_depth_valid & (~in_front)
    valid_candidate = alpha_valid & plane_depth_valid & in_front
    valid_partial = valid_candidate & (coverage < owner_coverage_min)
    valid_owner = valid_candidate & (coverage >= owner_coverage_min)

    classification_rgb = torch.zeros(
        (3, *accumulated_alpha.shape[-2:]),
        device=accumulated_alpha.device,
        dtype=accumulated_alpha.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(
            torch.where(mask.repeat(3, 1, 1), color_tensor, classification_rgb)
        )

    paint(low_alpha, (0.0, 0.35, 1.0))
    paint(invalid_plane_depth, (1.0, 0.0, 1.0))
    paint(behind_source, (1.0, 0.5, 0.0))
    paint(valid_partial, (0.0, 1.0, 1.0))
    paint(valid_owner, (0.0, 1.0, 0.0))
    return {
        "sand_failure_no_footprint": no_footprint,
        "sand_failure_low_alpha": low_alpha,
        "sand_failure_invalid_plane_depth": invalid_plane_depth,
        "sand_failure_behind_source": behind_source,
        "sand_failure_valid_partial": valid_partial,
        "sand_failure_valid_owner": valid_owner,
        "sand_failure_classification": classification_rgb,
    }


def classify_sand_depth_competition(
    depth_delta: torch.Tensor,
    comparison_valid: torch.Tensor,
    behind_source: torch.Tensor,
    coverage: torch.Tensor,
    *,
    depth_softness: float,
    interior_coverage_min: float = 0.95,
    interior_erode_kernel: int = 5,
) -> dict[str, torch.Tensor]:
    """Refine source-depth rejection without changing the owner decision.

    ``depth_delta = source_depth - sand_depth + bias`` is negative exactly
    where the source wins the discrete half-weight depth test.  The masks here
    answer two independent diagnostic questions:

    * Is the rejection within the configured soft-depth band (near-coplanar),
      or is the source substantially closer (strong occluder)?
    * Is it surrounded by densely covered sand in screen space, or does it lie
      near the projected sand silhouette?

    The interior mask is an erosion of high coverage, deliberately independent
    of the rejected pixels themselves.  Consequently an internal crack remains
    classified as interior instead of turning itself into a new boundary.
    """

    tensors = (comparison_valid, behind_source, coverage)
    if any(tensor.shape != depth_delta.shape for tensor in tensors):
        raise ValueError("all sand depth-competition inputs must match")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    interior_coverage_min = _finite_scalar(
        interior_coverage_min, "interior_coverage_min"
    )
    if depth_softness < 0.0:
        raise ValueError("depth_softness must be non-negative")
    if not 0.0 <= interior_coverage_min <= 1.0:
        raise ValueError("interior_coverage_min must lie in [0,1]")
    if interior_erode_kernel < 1 or interior_erode_kernel % 2 == 0:
        raise ValueError("interior_erode_kernel must be a positive odd integer")

    comparison_valid = comparison_valid.to(torch.bool)
    behind_source = behind_source.to(torch.bool) & comparison_valid
    high_coverage = coverage >= interior_coverage_min
    # Binary erosion via max-pooling the complement keeps this helper free of
    # extra image-processing dependencies and works identically on CPU/CUDA.
    high_coverage_holes = (~high_coverage).to(coverage.dtype)
    erosion_radius = interior_erode_kernel // 2
    # Treat pixels beyond the image as uncovered.  Zero-padding max-pool would
    # otherwise mislabel an image-border footprint as a sand interior.
    padded_holes = F.pad(
        high_coverage_holes[None, ...],
        (erosion_radius,) * 4,
        mode="constant",
        value=1.0,
    )
    neighborhood_has_hole = F.max_pool2d(
        padded_holes,
        kernel_size=interior_erode_kernel,
        stride=1,
        padding=0,
    )[0] > 0.0
    interior = high_coverage & (~neighborhood_has_hole)

    if depth_softness > 0.0:
        near_coplanar = behind_source & (depth_delta >= -depth_softness)
        normalization_scale = depth_softness
    else:
        # With a hard depth test there is no configured near-coplanar band.
        near_coplanar = torch.zeros_like(behind_source)
        normalization_scale = 1.0
    strong_occluder = behind_source & (~near_coplanar)
    boundary = behind_source & (~interior)
    interior_near = near_coplanar & interior
    interior_strong = strong_occluder & interior

    classification_rgb = torch.zeros(
        (3, *depth_delta.shape[-2:]),
        device=depth_delta.device,
        dtype=depth_delta.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(
            torch.where(mask.repeat(3, 1, 1), color_tensor, classification_rgb)
        )

    # Orange remains the generic/boundary failure color used by the preceding
    # classifier; yellow and red isolate the two interior cases of interest.
    paint(boundary, (1.0, 0.5, 0.0))
    paint(interior_near, (1.0, 1.0, 0.0))
    paint(interior_strong, (1.0, 0.0, 0.0))

    normalized_delta = torch.where(
        comparison_valid,
        depth_delta / normalization_scale,
        torch.zeros_like(depth_delta),
    )
    # Diverging diagnostic: red=source closer, white=near equal, blue=sand
    # closer. Saturation at +/-2 softness units keeps small biases visible.
    signed = torch.clamp(normalized_delta / 2.0, -1.0, 1.0)
    delta_rgb = torch.stack(
        (
            1.0 - torch.clamp(signed, min=0.0),
            1.0 - torch.abs(signed),
            1.0 - torch.clamp(-signed, min=0.0),
        ),
        dim=0,
    ).squeeze(1)
    delta_rgb = delta_rgb * comparison_valid.to(depth_delta.dtype)

    return {
        "sand_depth_comparison_valid": comparison_valid,
        "sand_depth_high_coverage_interior": interior,
        "sand_depth_behind_boundary": boundary,
        "sand_depth_behind_near_coplanar": near_coplanar,
        "sand_depth_behind_strong_occluder": strong_occluder,
        "sand_depth_interior_near_coplanar": interior_near,
        "sand_depth_interior_strong_occluder": interior_strong,
        "sand_depth_competition_classification": classification_rgb,
        "sand_depth_delta_normalized": normalized_delta,
        "sand_depth_delta_visualization": delta_rgb,
    }


def classify_sand_plane_depth_reference(
    plane_depth: torch.Tensor,
    center_depth: torch.Tensor,
    source_depth: torch.Tensor,
    valid_source: torch.Tensor,
    valid_sand: torch.Tensor,
    behind_source: torch.Tensor,
    high_coverage_interior: torch.Tensor,
    *,
    depth_bias: float,
    depth_softness: float,
) -> dict[str, torch.Tensor]:
    """Compare reconstructed sand-plane depth with center-depth splatting.

    Center depth uses the same Gaussian ordering, opacity and footprint as the
    plane pass, but alpha-composites each Gaussian center's camera-space z. It
    is not an exact mesh intersection; it is an independent, denominator-free
    reference that can reveal when a blended tangent plane flips the source vs
    sand owner decision.

    This function classifies only pixels already rejected as behind source and
    never changes the production depth or owner buffers.
    """

    tensors = (
        center_depth,
        source_depth,
        valid_source,
        valid_sand,
        behind_source,
        high_coverage_interior,
    )
    if any(tensor.shape != plane_depth.shape for tensor in tensors):
        raise ValueError("all sand plane-reference inputs must match")
    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    if depth_bias < 0.0 or depth_softness < 0.0:
        raise ValueError("depth bias and softness must be non-negative")

    valid_source = valid_source.to(torch.bool)
    valid_sand = valid_sand.to(torch.bool)
    behind_source = behind_source.to(torch.bool)
    high_coverage_interior = high_coverage_interior.to(torch.bool)
    center_valid = torch.isfinite(center_depth) & (center_depth > 0.0)
    reference_valid = valid_source & valid_sand & center_valid
    eligible = behind_source & reference_valid
    reference_invalid = behind_source & (~reference_valid)

    # Same sign convention as the production plane-depth delta: positive means
    # the center-depth reference places sand in front of the source scene.
    source_center_delta = source_depth - center_depth + depth_bias
    reference_sand_in_front = eligible & (source_center_delta >= 0.0)
    if depth_softness > 0.0:
        reference_near_behind = (
            eligible
            & (source_center_delta < 0.0)
            & (source_center_delta >= -depth_softness)
        )
        normalization_scale = depth_softness
    else:
        reference_near_behind = torch.zeros_like(eligible)
        normalization_scale = 1.0
    reference_strong_behind = (
        eligible & (~reference_sand_in_front) & (~reference_near_behind)
    )

    plane_source_delta = source_depth - plane_depth + depth_bias
    if depth_softness > 0.0:
        plane_strong_behind = eligible & (
            plane_source_delta < -depth_softness
        )
    else:
        plane_strong_behind = eligible & (plane_source_delta < 0.0)
    # Restrict the headline disagreement count to the original red class, not
    # the small near-coplanar (yellow) subset, so its reported denominator and
    # numerator describe exactly the same population.
    interior_disagreement = (
        reference_sand_in_front
        & high_coverage_interior
        & plane_strong_behind
    )
    interior_near_behind = reference_near_behind & high_coverage_interior
    interior_strong_behind = reference_strong_behind & high_coverage_interior

    classification_rgb = torch.zeros(
        (3, *plane_depth.shape[-2:]),
        device=plane_depth.device,
        dtype=plane_depth.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(
            torch.where(mask.repeat(3, 1, 1), color_tensor, classification_rgb)
        )

    paint(reference_invalid, (1.0, 0.0, 1.0))
    paint(reference_sand_in_front, (0.0, 1.0, 0.0))
    paint(reference_near_behind, (1.0, 1.0, 0.0))
    paint(reference_strong_behind, (1.0, 0.0, 0.0))

    plane_center_residual = torch.where(
        reference_valid,
        plane_depth - center_depth,
        torch.zeros_like(plane_depth),
    )
    normalized_residual = plane_center_residual / normalization_scale
    # Red means the reconstructed plane lies farther from the camera than its
    # center-depth reference; blue means it lies closer. White is agreement.
    signed = torch.clamp(normalized_residual / 2.0, -1.0, 1.0)
    residual_rgb = torch.stack(
        (
            1.0 - torch.clamp(-signed, min=0.0),
            1.0 - torch.abs(signed),
            1.0 - torch.clamp(signed, min=0.0),
        ),
        dim=0,
    ).squeeze(1)
    residual_rgb = residual_rgb * reference_valid.to(plane_depth.dtype)

    return {
        "sand_plane_reference_valid": reference_valid,
        "sand_plane_reference_invalid": reference_invalid,
        "sand_plane_reference_sand_in_front": reference_sand_in_front,
        "sand_plane_reference_near_behind": reference_near_behind,
        "sand_plane_reference_strong_behind": reference_strong_behind,
        "sand_plane_reference_interior_disagreement": interior_disagreement,
        "sand_plane_reference_interior_near_behind": interior_near_behind,
        "sand_plane_reference_interior_strong_behind": interior_strong_behind,
        "sand_plane_reference_classification": classification_rgb,
        "sand_plane_center_residual_raw": plane_center_residual,
        "sand_plane_center_residual_normalized": normalized_residual,
        "sand_plane_center_residual_visualization": residual_rgb,
        "sand_source_center_delta_raw": source_center_delta,
    }


def classify_sand_physical_reference_depth_loss(
    target_soft_coverage: torch.Tensor,
    gaussian_coverage: torch.Tensor,
    depth_in_front: torch.Tensor,
    center_depth: torch.Tensor,
    center_margin: torch.Tensor,
    triangle_reference_depth: torch.Tensor,
    triangle_reference_coverage: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    depth_bias: float,
    depth_softness: float,
    support_min: float = 0.02,
    rejected_in_front_max: float = 0.02,
    relative_margin: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Split ambiguous thin-sand loss with an independent physical z-buffer.

    The candidate population matches the continuous-RGB ``blue`` class used
    by the cohesion-tuning pixel audit: the simulated soft surface exists, a
    Gaussian footprint exists, but its source-depth weight has collapsed and
    the alpha-composited center-depth moments cannot confidently recover it.

    The triangle reference is rasterized directly from the simulated height
    field and therefore does not share the Gaussian plane/normal blending that
    caused the ambiguity.  Comparing that reference with source depth tells us
    whether the rejected pixel is likely real rear thin sand, a true source
    occlusion, near-coplanar, or lacks a usable physical reference.  This
    function is diagnostic only and never changes RGB, depth, or ownership.
    """

    tensors = (
        gaussian_coverage,
        depth_in_front,
        center_depth,
        center_margin,
        triangle_reference_depth,
        triangle_reference_coverage,
        source_depth,
        source_valid,
    )
    if any(tensor.shape != target_soft_coverage.shape for tensor in tensors):
        raise ValueError("all physical-reference depth-loss inputs must match")
    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    support_min = _finite_scalar(support_min, "support_min")
    rejected_in_front_max = _finite_scalar(
        rejected_in_front_max, "rejected_in_front_max"
    )
    relative_margin = _finite_scalar(relative_margin, "relative_margin")
    if depth_bias < 0.0 or depth_softness < 0.0 or relative_margin < 0.0:
        raise ValueError("physical-reference depth margins must be non-negative")
    if not 0.0 <= support_min <= 1.0:
        raise ValueError("support_min must lie in [0,1]")
    if not 0.0 <= rejected_in_front_max <= 1.0:
        raise ValueError("rejected_in_front_max must lie in [0,1]")

    source_valid = source_valid.to(torch.bool)
    triangle_reference_coverage = triangle_reference_coverage.to(torch.bool)
    physical_support = target_soft_coverage > support_min
    rasterized_support = gaussian_coverage > support_min
    fully_depth_rejected = depth_in_front < rejected_in_front_max

    center_source_delta = source_depth - center_depth + depth_bias
    center_numeric_valid = (
        torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_margin)
        & (center_margin >= 0.0)
    )
    center_confident_front = (
        source_valid
        & center_numeric_valid
        & (center_source_delta > center_margin)
    )
    candidate = (
        physical_support
        & rasterized_support
        & fully_depth_rejected
        & (~center_confident_front)
    )

    reference_valid = (
        triangle_reference_coverage
        & torch.isfinite(triangle_reference_depth)
        & (triangle_reference_depth > 0.0)
    )
    reference_margin = torch.maximum(
        torch.full_like(triangle_reference_depth, depth_softness),
        relative_margin * triangle_reference_depth,
    )
    triangle_source_delta = (
        source_depth - triangle_reference_depth + depth_bias
    )

    # Without a valid source surface the independently rasterized sand is the
    # only physical hit. Otherwise require a symmetric confidence interval so
    # near-coplanar ground does not get mislabeled as a safe recovery.
    reference_sand_front = candidate & reference_valid & (
        (~source_valid) | (triangle_source_delta > reference_margin)
    )
    source_front = (
        candidate
        & reference_valid
        & source_valid
        & (triangle_source_delta < -reference_margin)
    )
    near_coplanar = (
        candidate
        & reference_valid
        & source_valid
        & (~reference_sand_front)
        & (~source_front)
    )
    reference_invalid = candidate & (~reference_valid)

    classification_rgb = torch.zeros(
        (3, *target_soft_coverage.shape[-2:]),
        device=target_soft_coverage.device,
        dtype=target_soft_coverage.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(
            torch.where(mask.repeat(3, 1, 1), color_tensor, classification_rgb)
        )

    paint(reference_sand_front, (0.0, 1.0, 0.0))
    paint(source_front, (1.0, 0.0, 0.0))
    paint(near_coplanar, (1.0, 1.0, 0.0))
    paint(reference_invalid, (1.0, 0.0, 1.0))

    return {
        "sand_physical_reference_depth_loss_candidate": candidate,
        "sand_physical_reference_sand_front": reference_sand_front,
        "sand_physical_reference_source_front": source_front,
        "sand_physical_reference_near_coplanar": near_coplanar,
        "sand_physical_reference_invalid": reference_invalid,
        "sand_physical_reference_center_confident_front": (
            center_confident_front
        ),
        "sand_physical_reference_depth_loss_classification": (
            classification_rgb
        ),
        "sand_physical_reference_source_delta_raw": torch.where(
            reference_valid & source_valid,
            triangle_source_delta,
            torch.zeros_like(triangle_source_delta),
        ),
        "sand_physical_reference_margin_raw": torch.where(
            reference_valid,
            reference_margin,
            torch.zeros_like(reference_margin),
        ),
    }


def classify_sand_center_trusted_invalid_depth(
    target_soft_coverage: torch.Tensor,
    gaussian_coverage: torch.Tensor,
    resolved_depth_valid: torch.Tensor,
    center_depth: torch.Tensor,
    center_margin: torch.Tensor,
    center_reference_reliable: torch.Tensor,
    triangle_reference_depth: torch.Tensor,
    triangle_reference_coverage: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    *,
    depth_bias: float,
    depth_softness: float,
    physical_core_min: float = 0.5,
    owner_coverage_min: float = 0.5,
    relative_margin: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Classify core holes whose center order is confident but plane depth is not.

    This population is deliberately disjoint from the earlier ambiguous-center
    recovery.  It requires enough physical and Gaussian coverage to become a
    discrete owner, an invalid *resolved* plane depth after the existing
    stability fallback, and a center estimate whose source-order margin says
    sand is in front. Center variance reliability is reported separately: a
    large source-order margin does not imply a low-variance center estimate.
    The independent triangle z-buffer then decides whether the remaining hole
    is safe sand, a true source occlusion, near-coplanar, or has no usable
    physical reference.  The result is diagnostic only.
    """

    tensors = (
        gaussian_coverage,
        resolved_depth_valid,
        center_depth,
        center_margin,
        center_reference_reliable,
        triangle_reference_depth,
        triangle_reference_coverage,
        source_depth,
        source_valid,
    )
    if any(tensor.shape != target_soft_coverage.shape for tensor in tensors):
        raise ValueError("all center-trusted invalid-depth inputs must match")
    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    physical_core_min = _finite_scalar(physical_core_min, "physical_core_min")
    owner_coverage_min = _finite_scalar(
        owner_coverage_min, "owner_coverage_min"
    )
    relative_margin = _finite_scalar(relative_margin, "relative_margin")
    if depth_bias < 0.0 or depth_softness < 0.0 or relative_margin < 0.0:
        raise ValueError("center-trusted depth margins must be non-negative")
    if not 0.0 <= physical_core_min <= 1.0:
        raise ValueError("physical_core_min must lie in [0,1]")
    if not 0.0 <= owner_coverage_min <= 1.0:
        raise ValueError("owner_coverage_min must lie in [0,1]")

    source_valid = source_valid.to(torch.bool)
    resolved_depth_valid = resolved_depth_valid.to(torch.bool)
    center_reference_reliable = center_reference_reliable.to(torch.bool)
    triangle_reference_coverage = triangle_reference_coverage.to(torch.bool)
    physical_core = target_soft_coverage >= physical_core_min
    owner_support = gaussian_coverage >= owner_coverage_min
    resolved_invalid = ~resolved_depth_valid

    center_numeric_valid = (
        torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_margin)
        & (center_margin >= 0.0)
    )
    center_source_delta = source_depth - center_depth + depth_bias
    center_confident_front = (
        source_valid
        & center_numeric_valid
        & (center_source_delta > center_margin)
    )
    candidate = (
        physical_core
        & owner_support
        & resolved_invalid
        & center_confident_front
    )

    reference_valid = (
        triangle_reference_coverage
        & torch.isfinite(triangle_reference_depth)
        & (triangle_reference_depth > 0.0)
    )
    reference_margin = torch.maximum(
        torch.full_like(triangle_reference_depth, depth_softness),
        relative_margin * triangle_reference_depth,
    )
    triangle_source_delta = source_depth - triangle_reference_depth + depth_bias
    triangle_sand_front = candidate & reference_valid & (
        triangle_source_delta > reference_margin
    )
    source_front = candidate & reference_valid & (
        triangle_source_delta < -reference_margin
    )
    near_coplanar = (
        candidate
        & reference_valid
        & (~triangle_sand_front)
        & (~source_front)
    )
    reference_invalid = candidate & (~reference_valid)
    candidate_center_reliable = candidate & center_reference_reliable
    candidate_center_unreliable = candidate & (~center_reference_reliable)

    classification_rgb = torch.zeros(
        (3, *target_soft_coverage.shape[-2:]),
        device=target_soft_coverage.device,
        dtype=target_soft_coverage.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(torch.where(
            mask.repeat(3, 1, 1), color_tensor, classification_rgb
        ))

    paint(triangle_sand_front, (0.0, 1.0, 0.0))
    paint(source_front, (1.0, 0.0, 0.0))
    paint(near_coplanar, (1.0, 1.0, 0.0))
    paint(reference_invalid, (1.0, 0.0, 1.0))

    return {
        "sand_center_trusted_invalid_candidate": candidate,
        "sand_center_trusted_invalid_triangle_sand_front": (
            triangle_sand_front
        ),
        "sand_center_trusted_invalid_source_front": source_front,
        "sand_center_trusted_invalid_near_coplanar": near_coplanar,
        "sand_center_trusted_invalid_reference_invalid": reference_invalid,
        "sand_center_trusted_invalid_classification": classification_rgb,
        "sand_center_trusted_invalid_physical_core": physical_core,
        "sand_center_trusted_invalid_owner_support": owner_support,
        "sand_center_trusted_invalid_center_confident_front": (
            center_confident_front
        ),
        "sand_center_trusted_invalid_center_reliable": (
            candidate_center_reliable
        ),
        "sand_center_trusted_invalid_center_unreliable": (
            candidate_center_unreliable
        ),
        "sand_center_trusted_invalid_center_source_delta_raw": torch.where(
            candidate,
            center_source_delta,
            torch.zeros_like(center_source_delta),
        ),
        "sand_center_trusted_invalid_triangle_source_delta_raw": torch.where(
            candidate & reference_valid,
            triangle_source_delta,
            torch.zeros_like(triangle_source_delta),
        ),
        "sand_center_trusted_invalid_reference_margin_raw": torch.where(
            candidate & reference_valid,
            reference_margin,
            torch.zeros_like(reference_margin),
        ),
    }


def resolve_sand_center_trusted_invalid_depth(
    current_depth: torch.Tensor,
    triangle_reference_depth: torch.Tensor,
    triangle_sand_front: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Recover only core invalid holes independently confirmed by the mesh.

    The center-depth estimate participates only in the upstream candidate
    diagnosis; its high-variance value is never written here.  Production
    depth always comes from the direct simulated-triangle z-buffer, and the
    caller must recompute visibility/ownership from the returned depth.
    """

    if triangle_reference_depth.shape != current_depth.shape:
        raise ValueError("center-trusted triangle depth must match current depth")
    if triangle_sand_front.shape != current_depth.shape:
        raise ValueError("center-trusted recovery mask must match current depth")
    triangle_sand_front = triangle_sand_front.to(torch.bool)
    reference_valid = (
        torch.isfinite(triangle_reference_depth)
        & (triangle_reference_depth > 0.0)
    )
    candidate = triangle_sand_front & reference_valid
    used = candidate if bool(enabled) else torch.zeros_like(candidate)
    resolved_depth = torch.where(
        used, triangle_reference_depth, current_depth
    )
    return {
        "sand_center_trusted_invalid_recovery_candidate": candidate,
        "sand_center_trusted_invalid_recovery_used": used,
        "sand_center_trusted_invalid_recovery_resolved_depth": resolved_depth,
    }


def resolve_sand_center_trusted_invalid_normal(
    blended_normal_world: torch.Tensor,
    triangle_normal_world: torch.Tensor,
    depth_recovery_used: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Resolve macro normals only where the second depth recovery was used.

    The recovered camera depth comes from the independently rasterized sand
    triangle, so retaining a foreground/background alpha-blended Gaussian
    normal at the same pixel is internally inconsistent. Align the triangle
    winding to the blended normal's hemisphere and, only when explicitly
    enabled, use that face normal on the exact depth-recovery mask. No mask
    dilation or confidence relaxation is performed here.
    """

    if blended_normal_world.ndim != 3 or blended_normal_world.shape[0] != 3:
        raise ValueError("blended_normal_world must have shape [3,H,W]")
    if triangle_normal_world.shape != blended_normal_world.shape:
        raise ValueError("triangle_normal_world must match blended normal")
    expected_mask_shape = (1,) + tuple(blended_normal_world.shape[1:])
    if depth_recovery_used.shape != expected_mask_shape:
        raise ValueError("depth_recovery_used must have shape [1,H,W]")
    if not torch.isfinite(blended_normal_world).all():
        raise ValueError("blended normal must contain only finite values")
    if not torch.isfinite(triangle_normal_world).all():
        raise ValueError("triangle normal must contain only finite values")

    candidate = depth_recovery_used.to(torch.bool)
    aligned_triangle_normal, angle = align_reference_normals(
        triangle_normal_world,
        blended_normal_world,
        candidate,
    )
    used = candidate if bool(enabled) else torch.zeros_like(candidate)
    resolved_normal = torch.where(
        used.repeat(3, 1, 1),
        aligned_triangle_normal,
        blended_normal_world,
    )
    return {
        "sand_center_trusted_invalid_normal_recovery_candidate": candidate,
        "sand_center_trusted_invalid_normal_recovery_used": used,
        "sand_center_trusted_invalid_normal_recovery_resolved": (
            resolved_normal
        ),
        "sand_center_trusted_invalid_normal_recovery_angle_degrees_raw": (
            angle
        ),
        "sand_center_trusted_invalid_normal_recovery_absolute_difference": (
            torch.abs(resolved_normal - blended_normal_world)
            * used.to(blended_normal_world.dtype)
        ),
        "sand_center_trusted_invalid_normal_recovery_enabled": bool(enabled),
    }


def resolve_sand_physical_reference_normal(
    blended_normal_world: torch.Tensor,
    triangle_normal_world: torch.Tensor,
    depth_recovery_used: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Resolve macro normals only where the first depth recovery was used.

    This deliberately shares the same winding alignment and exact-mask
    semantics as the second recovery. Separate output names and switches keep
    their different coverage populations independently auditable.
    """

    resolved = resolve_sand_center_trusted_invalid_normal(
        blended_normal_world,
        triangle_normal_world,
        depth_recovery_used,
        enabled=enabled,
    )
    return {
        "sand_physical_reference_normal_recovery_candidate": resolved[
            "sand_center_trusted_invalid_normal_recovery_candidate"
        ],
        "sand_physical_reference_normal_recovery_used": resolved[
            "sand_center_trusted_invalid_normal_recovery_used"
        ],
        "sand_physical_reference_normal_recovery_resolved": resolved[
            "sand_center_trusted_invalid_normal_recovery_resolved"
        ],
        "sand_physical_reference_normal_recovery_angle_degrees_raw": resolved[
            "sand_center_trusted_invalid_normal_recovery_angle_degrees_raw"
        ],
        "sand_physical_reference_normal_recovery_absolute_difference": resolved[
            "sand_center_trusted_invalid_normal_recovery_absolute_difference"
        ],
        "sand_physical_reference_normal_recovery_enabled": bool(enabled),
    }


def select_sand_recovery_attribute_region(
    physical_recovery_used: torch.Tensor,
    center_recovery_used: torch.Tensor,
    texture_valid: torch.Tensor,
    triangle_reference_coverage: torch.Tensor,
    *,
    physical_enabled: bool,
    center_enabled: bool,
) -> dict[str, object]:
    """Select exactly one conservative recovery for attribute diagnosis.

    The two depth recoveries address disjoint confidence classes, but their
    counterfactual appearance audit is otherwise identical. Keep the selector
    explicit so a diagnostic run cannot silently combine their masks and make
    coverage-stratified statistics impossible to interpret.
    """

    tensors = (
        center_recovery_used,
        texture_valid,
        triangle_reference_coverage,
    )
    if any(tensor.shape != physical_recovery_used.shape for tensor in tensors):
        raise ValueError("all recovery attribute masks must match")
    if bool(physical_enabled) and bool(center_enabled):
        raise ValueError(
            "physical and center recovery attribute diagnostics are mutually "
            "exclusive"
        )
    if not bool(physical_enabled) and not bool(center_enabled):
        raise ValueError("one recovery attribute diagnostic must be enabled")

    source = (
        physical_recovery_used
        if bool(physical_enabled)
        else center_recovery_used
    )
    region = (
        source.to(torch.bool)
        & texture_valid.to(torch.bool)
        & triangle_reference_coverage.to(torch.bool)
    )
    return {
        "sand_recovery_attribute_region_mask": region,
        "sand_recovery_attribute_scope": (
            "physical_reference" if bool(physical_enabled) else "center_trusted_invalid"
        ),
    }


def resolve_sand_physical_reference_depth(
    current_depth: torch.Tensor,
    triangle_reference_depth: torch.Tensor,
    reference_sand_front: torch.Tensor,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Replace only independently confirmed false-negative sand depths.

    ``reference_sand_front`` must come from
    :func:`classify_sand_physical_reference_depth_loss`, which has already
    excluded true source occlusions, near-coplanar comparisons, invalid mesh
    references, and center-depth cases handled by the older fallback. Keeping
    this final selector separate makes the behavior switch auditable: disabled
    runs return the original tensor bit-for-bit, while enabled runs change no
    pixel outside the conservative green diagnostic class.
    """

    if triangle_reference_depth.shape != current_depth.shape:
        raise ValueError("physical reference depth must match current depth")
    if reference_sand_front.shape != current_depth.shape:
        raise ValueError("physical reference recovery mask must match depth")
    reference_sand_front = reference_sand_front.to(torch.bool)
    reference_valid = (
        torch.isfinite(triangle_reference_depth)
        & (triangle_reference_depth > 0.0)
    )
    candidate = reference_sand_front & reference_valid
    used = candidate if bool(enabled) else torch.zeros_like(candidate)
    resolved_depth = torch.where(
        used, triangle_reference_depth, current_depth
    )
    return {
        "sand_physical_reference_recovery_candidate": candidate,
        "sand_physical_reference_recovery_used": used,
        "sand_physical_reference_recovery_resolved_depth": resolved_depth,
    }


def classify_sand_invalid_plane_depth(
    invalid_plane_depth: torch.Tensor,
    coverage: torch.Tensor,
    center_depth: torch.Tensor,
    center_depth_std: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    mean_normal_camera: torch.Tensor,
    normal_second_moments: torch.Tensor,
    camera_ray_forward: torch.Tensor,
    primitive_numeric_bad_fraction: torch.Tensor,
    primitive_orientation_bad_fraction: torch.Tensor,
    *,
    depth_bias: float,
    depth_softness: float,
    coverage_min: float = 0.95,
    erode_kernel: int = 5,
    primitive_bad_fraction_max: float = 0.01,
    incidence_epsilon: float = 0.02,
    incidence_std_max: float = 0.02,
    normal_variance_max: float = 0.02,
    center_std_relative_max: float = 0.01,
    recovery_sigma_scale: float = 3.0,
    recovery_relative_margin: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Split invalid sand plane depth into mutually exclusive causes.

    The plane rasterizer evaluates ``P / (-N.r + eps)`` after alpha blending
    per-Gaussian normals and plane distances.  This diagnostic separates an
    invalid result caused by a stable grazing denominator from cancellation
    between otherwise valid planes and from malformed/back-facing primitive
    inputs.  A second classification asks whether the independent center-depth
    reference could conservatively recover the pixel.  Neither classification
    modifies production depth or ownership.

    ``normal_second_moments`` stores ``[xx, yy, zz, xy, xz, yz]`` after the
    same alpha compositing and de-premultiplication as ``mean_normal_camera``.
    """

    scalar_tensors = (
        coverage,
        center_depth,
        center_depth_std,
        source_depth,
        source_valid,
        primitive_numeric_bad_fraction,
        primitive_orientation_bad_fraction,
    )
    if any(tensor.shape != invalid_plane_depth.shape for tensor in scalar_tensors):
        raise ValueError("all invalid-depth scalar inputs must match")
    expected_vector_shape = (3, *invalid_plane_depth.shape[-2:])
    expected_moment_shape = (6, *invalid_plane_depth.shape[-2:])
    if mean_normal_camera.shape != expected_vector_shape:
        raise ValueError("mean_normal_camera must have shape [3,H,W]")
    if camera_ray_forward.shape != expected_vector_shape:
        raise ValueError("camera_ray_forward must have shape [3,H,W]")
    if normal_second_moments.shape != expected_moment_shape:
        raise ValueError("normal_second_moments must have shape [6,H,W]")

    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    coverage_min = _finite_scalar(coverage_min, "coverage_min")
    primitive_bad_fraction_max = _finite_scalar(
        primitive_bad_fraction_max, "primitive_bad_fraction_max"
    )
    incidence_epsilon = _finite_scalar(incidence_epsilon, "incidence_epsilon")
    incidence_std_max = _finite_scalar(incidence_std_max, "incidence_std_max")
    normal_variance_max = _finite_scalar(
        normal_variance_max, "normal_variance_max"
    )
    center_std_relative_max = _finite_scalar(
        center_std_relative_max, "center_std_relative_max"
    )
    recovery_sigma_scale = _finite_scalar(
        recovery_sigma_scale, "recovery_sigma_scale"
    )
    recovery_relative_margin = _finite_scalar(
        recovery_relative_margin, "recovery_relative_margin"
    )
    if depth_bias < 0.0 or depth_softness < 0.0:
        raise ValueError("depth bias and softness must be non-negative")
    for name, value in (
        ("coverage_min", coverage_min),
        ("primitive_bad_fraction_max", primitive_bad_fraction_max),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    if any(
        value < 0.0
        for value in (
            incidence_epsilon,
            incidence_std_max,
            normal_variance_max,
            center_std_relative_max,
            recovery_sigma_scale,
            recovery_relative_margin,
        )
    ):
        raise ValueError("invalid-depth diagnostic thresholds must be non-negative")
    if erode_kernel < 1 or erode_kernel % 2 == 0:
        raise ValueError("erode_kernel must be a positive odd integer")

    invalid_plane_depth = invalid_plane_depth.to(torch.bool)
    source_valid = source_valid.to(torch.bool)
    high_coverage = coverage >= coverage_min
    erosion_radius = erode_kernel // 2
    padded_holes = F.pad(
        (~high_coverage).to(coverage.dtype)[None, ...],
        (erosion_radius,) * 4,
        mode="constant",
        value=1.0,
    )
    interior = high_coverage & (
        F.max_pool2d(
            padded_holes,
            kernel_size=erode_kernel,
            stride=1,
            padding=0,
        )[0]
        <= 0.0
    )
    boundary = invalid_plane_depth & (~interior)
    interior_invalid = invalid_plane_depth & interior

    ray = F.normalize(camera_ray_forward, dim=0, eps=1.0e-6)
    normal_mean_norm = torch.linalg.vector_norm(
        mean_normal_camera, dim=0, keepdim=True
    )
    normal_direction = mean_normal_camera / torch.clamp(
        normal_mean_norm, min=1.0e-6
    )
    # Sign is retained for raw inspection, while the cause test uses the
    # absolute incidence because both near-parallel orientations are singular.
    signed_incidence = -torch.sum(normal_direction * ray, dim=0, keepdim=True)
    incidence_abs = torch.abs(signed_incidence)

    xx, yy, zz, xy, xz, yz = normal_second_moments
    rx, ry, rz = ray
    incidence_second = (
        rx * rx * xx
        + ry * ry * yy
        + rz * rz * zz
        + 2.0 * rx * ry * xy
        + 2.0 * rx * rz * xz
        + 2.0 * ry * rz * yz
    )[None, ...]
    # E[(n.r)^2] - E[n.r]^2 detects cancellation even when E[n].r is close
    # to zero.  Trace covariance gives a ray-independent mixing indicator.
    mean_dot_ray = torch.sum(mean_normal_camera * ray, dim=0, keepdim=True)
    incidence_variance = torch.clamp(
        incidence_second - mean_dot_ray * mean_dot_ray, min=0.0
    )
    incidence_std = torch.sqrt(incidence_variance)
    normal_second_trace = (xx + yy + zz)[None, ...]
    normal_variance = torch.clamp(
        normal_second_trace - normal_mean_norm * normal_mean_norm, min=0.0
    )

    numeric_bad = interior_invalid & (
        primitive_numeric_bad_fraction > primitive_bad_fraction_max
    )
    orientation_bad = (
        interior_invalid
        & (~numeric_bad)
        & (
            primitive_orientation_bad_fraction
            > primitive_bad_fraction_max
        )
    )
    primitive_bad = numeric_bad | orientation_bad
    coherent = (
        (incidence_std <= incidence_std_max)
        & (normal_variance <= normal_variance_max)
    )
    stable_singular = (
        interior_invalid
        & (~primitive_bad)
        & (incidence_abs <= incidence_epsilon)
        & coherent
    )

    center_numeric_valid = (
        torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_depth_std)
        & (center_depth_std >= 0.0)
    )
    softness_tensor = torch.full_like(center_depth, depth_softness)
    center_uncertainty_limit = torch.maximum(
        softness_tensor, center_std_relative_max * center_depth
    )
    center_unstable = center_numeric_valid & (
        center_depth_std > center_uncertainty_limit
    )
    mixing_evidence = (
        (incidence_std > incidence_std_max)
        | (normal_variance > normal_variance_max)
        | center_unstable
    )
    mixed_planes = (
        interior_invalid
        & (~primitive_bad)
        & (~stable_singular)
        & mixing_evidence
    )
    unclassified = (
        interior_invalid
        & (~primitive_bad)
        & (~stable_singular)
        & (~mixed_planes)
    )

    cause_rgb = torch.zeros(
        (3, *invalid_plane_depth.shape[-2:]),
        device=coverage.device,
        dtype=coverage.dtype,
    )

    def paint(
        image: torch.Tensor,
        mask: torch.Tensor,
        color: tuple[float, float, float],
    ) -> None:
        color_tensor = torch.tensor(
            color, device=image.device, dtype=image.dtype
        )[:, None, None]
        image.copy_(torch.where(mask.repeat(3, 1, 1), color_tensor, image))

    paint(cause_rgb, boundary, (1.0, 0.5, 0.0))
    paint(cause_rgb, numeric_bad, (1.0, 0.0, 1.0))
    paint(cause_rgb, orientation_bad, (0.0, 0.35, 1.0))
    paint(cause_rgb, stable_singular, (1.0, 1.0, 0.0))
    paint(cause_rgb, mixed_planes, (1.0, 0.0, 0.0))
    paint(cause_rgb, unclassified, (0.0, 1.0, 1.0))

    center_reference_reliable = center_numeric_valid & (
        center_depth_std <= center_uncertainty_limit
    )
    recovery_margin = torch.maximum(
        softness_tensor,
        torch.maximum(
            recovery_sigma_scale * center_depth_std,
            recovery_relative_margin * center_depth,
        ),
    )
    recovery_margin = torch.where(
        center_numeric_valid,
        recovery_margin,
        torch.full_like(recovery_margin, float("inf")),
    )
    source_center_delta = source_depth - center_depth + depth_bias
    recovery_comparable = (
        interior_invalid & source_valid & center_reference_reliable
    )
    recovery_center_behind = recovery_comparable & (source_center_delta <= 0.0)
    recovery_weak_front = (
        recovery_comparable
        & (source_center_delta > 0.0)
        & (source_center_delta <= recovery_margin)
    )
    recovery_confident_front = recovery_comparable & (
        source_center_delta > recovery_margin
    )
    recovery_uncertain = interior_invalid & (~recovery_comparable)

    recovery_rgb = torch.zeros_like(cause_rgb)
    paint(recovery_rgb, boundary, (1.0, 0.5, 0.0))
    paint(recovery_rgb, recovery_uncertain, (1.0, 0.0, 1.0))
    paint(recovery_rgb, recovery_center_behind, (1.0, 0.0, 0.0))
    paint(recovery_rgb, recovery_weak_front, (1.0, 1.0, 0.0))
    paint(recovery_rgb, recovery_confident_front, (0.0, 1.0, 0.0))

    return {
        "sand_invalid_depth_boundary": boundary,
        "sand_invalid_depth_primitive_numeric": numeric_bad,
        "sand_invalid_depth_primitive_orientation": orientation_bad,
        "sand_invalid_depth_stable_singular": stable_singular,
        "sand_invalid_depth_mixed_planes": mixed_planes,
        "sand_invalid_depth_unclassified": unclassified,
        "sand_invalid_depth_cause_classification": cause_rgb,
        "sand_invalid_depth_recovery_uncertain": recovery_uncertain,
        "sand_invalid_depth_recovery_center_behind": recovery_center_behind,
        "sand_invalid_depth_recovery_weak_front": recovery_weak_front,
        "sand_invalid_depth_recovery_confident_front": (
            recovery_confident_front
        ),
        "sand_invalid_depth_recovery_classification": recovery_rgb,
        "sand_invalid_depth_interior": interior,
        "sand_invalid_depth_incidence_signed_raw": signed_incidence,
        "sand_invalid_depth_incidence_abs_raw": incidence_abs,
        "sand_invalid_depth_incidence_std_raw": incidence_std,
        "sand_invalid_depth_normal_variance_raw": normal_variance,
        "sand_invalid_depth_primitive_numeric_fraction_raw": (
            primitive_numeric_bad_fraction
        ),
        "sand_invalid_depth_primitive_orientation_fraction_raw": (
            primitive_orientation_bad_fraction
        ),
        "sand_invalid_depth_recovery_margin_raw": recovery_margin,
    }


def resolve_sand_plane_depth_stability(
    plane_depth: torch.Tensor,
    center_depth: torch.Tensor,
    center_depth_std: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    coverage: torch.Tensor,
    *,
    enabled: bool,
    depth_bias: float,
    depth_softness: float,
    coverage_min: float = 0.95,
    erode_kernel: int = 5,
    sigma_scale: float = 3.0,
    relative_margin: float = 0.002,
    center_std_relative_max: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Conservatively resolve contradictory sand plane depths.

    A fallback is allowed only when a dense interior pixel has a reliable
    center-depth reference and the two depth estimates confidently reverse the
    source/sand owner decision. Merely disagreeing in magnitude is not enough.
    The returned masks are useful even with ``enabled=False`` for strict A/B
    diagnosis; only ``used`` and ``resolved_depth`` depend on the switch.
    """

    tensors = (center_depth, center_depth_std, source_depth, source_valid, coverage)
    if any(tensor.shape != plane_depth.shape for tensor in tensors):
        raise ValueError("all sand stability-fallback inputs must match")
    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    coverage_min = _finite_scalar(coverage_min, "coverage_min")
    sigma_scale = _finite_scalar(sigma_scale, "sigma_scale")
    relative_margin = _finite_scalar(relative_margin, "relative_margin")
    center_std_relative_max = _finite_scalar(
        center_std_relative_max, "center_std_relative_max"
    )
    if depth_bias < 0.0 or depth_softness < 0.0:
        raise ValueError("depth bias and softness must be non-negative")
    if not 0.0 <= coverage_min <= 1.0:
        raise ValueError("coverage_min must lie in [0,1]")
    if sigma_scale < 0.0 or relative_margin < 0.0:
        raise ValueError("fallback margin scales must be non-negative")
    if center_std_relative_max < 0.0:
        raise ValueError("center_std_relative_max must be non-negative")
    if erode_kernel < 1 or erode_kernel % 2 == 0:
        raise ValueError("erode_kernel must be a positive odd integer")

    source_valid = source_valid.to(torch.bool)
    plane_valid = torch.isfinite(plane_depth) & (plane_depth > 0.0)
    center_valid = torch.isfinite(center_depth) & (center_depth > 0.0)
    center_std_valid = torch.isfinite(center_depth_std) & (center_depth_std >= 0.0)

    high_coverage = coverage >= coverage_min
    erosion_radius = erode_kernel // 2
    padded_holes = F.pad(
        (~high_coverage).to(coverage.dtype)[None, ...],
        (erosion_radius,) * 4,
        mode="constant",
        value=1.0,
    )
    interior = high_coverage & (
        F.max_pool2d(
            padded_holes,
            kernel_size=erode_kernel,
            stride=1,
            padding=0,
        )[0]
        <= 0.0
    )

    softness_tensor = torch.full_like(center_depth, depth_softness)
    base_margin = torch.maximum(
        softness_tensor,
        relative_margin * center_depth,
    )
    margin = torch.maximum(
        softness_tensor,
        torch.maximum(
            sigma_scale * center_depth_std,
            relative_margin * center_depth,
        ),
    )
    reference_numeric_valid = center_valid & center_std_valid
    margin = torch.where(
        reference_numeric_valid,
        margin,
        torch.full_like(margin, float("inf")),
    )
    uncertainty_limit = torch.maximum(
        softness_tensor,
        center_std_relative_max * center_depth,
    )
    reference_reliable = (
        reference_numeric_valid & (center_depth_std <= uncertainty_limit)
    )

    plane_source_delta = source_depth - plane_depth + depth_bias
    center_source_delta = source_depth - center_depth + depth_bias
    plane_rejects_base = (~plane_valid) | (
        plane_source_delta < -base_margin
    )
    center_in_front_base = center_source_delta > base_margin
    candidate = (
        source_valid
        & reference_numeric_valid
        & plane_rejects_base
        & center_in_front_base
    )
    rejected_by_boundary = candidate & (~interior)
    rejected_by_uncertain_center = candidate & interior & (~reference_reliable)
    passes_adaptive_margin = (
        ((~plane_valid) | (plane_source_delta < -margin))
        & (center_source_delta > margin)
    )
    rejected_by_adaptive_margin = (
        candidate
        & interior
        & reference_reliable
        & (~passes_adaptive_margin)
    )
    eligible = (
        candidate
        & interior
        & reference_reliable
        & passes_adaptive_margin
    )
    used = eligible if bool(enabled) else torch.zeros_like(eligible)
    resolved_depth = torch.where(used, center_depth, plane_depth)

    return {
        "sand_plane_fallback_candidate": candidate,
        "sand_plane_fallback_used": used,
        "sand_plane_fallback_rejected_by_boundary": rejected_by_boundary,
        "sand_plane_fallback_rejected_by_uncertain_center": (
            rejected_by_uncertain_center
        ),
        "sand_plane_fallback_rejected_by_adaptive_margin": (
            rejected_by_adaptive_margin
        ),
        "sand_plane_fallback_reference_reliable": reference_reliable,
        "sand_plane_fallback_interior": interior,
        "sand_plane_fallback_margin_raw": margin,
        "sand_plane_fallback_center_std_raw": center_depth_std,
        "sand_plane_fallback_resolved_depth": resolved_depth,
    }


def resolve_sand_oriented_plane_depth(
    plane_depth: torch.Tensor,
    invalid_plane_depth: torch.Tensor,
    oriented_plane_depth: torch.Tensor,
    center_depth: torch.Tensor,
    center_depth_std: torch.Tensor,
    source_depth: torch.Tensor,
    source_valid: torch.Tensor,
    coverage: torch.Tensor,
    *,
    enabled: bool,
    depth_bias: float,
    depth_softness: float,
    coverage_min: float = 0.95,
    erode_kernel: int = 5,
    sigma_scale: float = 3.0,
    relative_margin: float = 0.002,
    center_std_relative_max: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Recover only well-supported invalid depths with an oriented candidate.

    Camera-facing primitive normals make ``d / (-n.r)`` positive, but that is
    not sufficient evidence that a boundary splat represents a visible sand
    surface.  The oriented depth is therefore a candidate, never a global
    replacement.  Recovery is restricted to the original invalid population,
    an eroded high-coverage interior, a reliable center-depth reference,
    candidate/reference agreement and a shared safety margin in front of the
    source surface.  In particular, this function never changes a legacy
    ``behind_source`` decision whose plane depth was already valid.
    """

    tensors = (
        invalid_plane_depth,
        oriented_plane_depth,
        center_depth,
        center_depth_std,
        source_depth,
        source_valid,
        coverage,
    )
    if any(tensor.shape != plane_depth.shape for tensor in tensors):
        raise ValueError("all oriented-depth recovery inputs must match")
    depth_bias = _finite_scalar(depth_bias, "depth_bias")
    depth_softness = _finite_scalar(depth_softness, "depth_softness")
    coverage_min = _finite_scalar(coverage_min, "coverage_min")
    sigma_scale = _finite_scalar(sigma_scale, "sigma_scale")
    relative_margin = _finite_scalar(relative_margin, "relative_margin")
    center_std_relative_max = _finite_scalar(
        center_std_relative_max, "center_std_relative_max"
    )
    if depth_bias < 0.0 or depth_softness < 0.0:
        raise ValueError("depth bias and softness must be non-negative")
    if not 0.0 <= coverage_min <= 1.0:
        raise ValueError("coverage_min must lie in [0,1]")
    if any(
        value < 0.0
        for value in (
            sigma_scale,
            relative_margin,
            center_std_relative_max,
        )
    ):
        raise ValueError("oriented-depth recovery thresholds must be non-negative")
    if erode_kernel < 1 or erode_kernel % 2 == 0:
        raise ValueError("erode_kernel must be a positive odd integer")

    invalid_plane_depth = invalid_plane_depth.to(torch.bool)
    source_valid = source_valid.to(torch.bool)
    oriented_valid = torch.isfinite(oriented_plane_depth) & (
        oriented_plane_depth > 0.0
    )
    center_numeric_valid = (
        torch.isfinite(center_depth)
        & (center_depth > 0.0)
        & torch.isfinite(center_depth_std)
        & (center_depth_std >= 0.0)
    )

    high_coverage = coverage >= coverage_min
    erosion_radius = erode_kernel // 2
    padded_holes = F.pad(
        (~high_coverage).to(coverage.dtype)[None, ...],
        (erosion_radius,) * 4,
        mode="constant",
        value=1.0,
    )
    interior = high_coverage & (
        F.max_pool2d(
            padded_holes,
            kernel_size=erode_kernel,
            stride=1,
            padding=0,
        )[0]
        <= 0.0
    )

    softness_tensor = torch.full_like(center_depth, depth_softness)
    uncertainty_limit = torch.maximum(
        softness_tensor,
        center_std_relative_max * center_depth,
    )
    center_reliable = center_numeric_valid & (
        center_depth_std <= uncertainty_limit
    )
    margin = torch.maximum(
        softness_tensor,
        torch.maximum(
            sigma_scale * center_depth_std,
            relative_margin * center_depth,
        ),
    )
    margin = torch.where(
        center_numeric_valid,
        margin,
        torch.full_like(margin, float("inf")),
    )

    boundary = invalid_plane_depth & (~interior)
    interior_invalid = invalid_plane_depth & interior
    candidate_invalid = interior_invalid & (~oriented_valid)
    uncertain_reference = (
        interior_invalid
        & oriented_valid
        & ((~source_valid) | (~center_reliable))
    )
    comparable = (
        interior_invalid & oriented_valid & source_valid & center_reliable
    )
    candidate_center_residual = oriented_plane_depth - center_depth
    agrees_with_center = torch.abs(candidate_center_residual) <= margin
    depth_disagreement = comparable & (~agrees_with_center)
    candidate_source_delta = (
        source_depth - oriented_plane_depth + depth_bias
    )
    center_source_delta = source_depth - center_depth + depth_bias
    safely_in_front = (
        (candidate_source_delta > margin)
        & (center_source_delta > margin)
    )
    insufficient_front_margin = (
        comparable & agrees_with_center & (~safely_in_front)
    )
    eligible = comparable & agrees_with_center & safely_in_front
    used = eligible if bool(enabled) else torch.zeros_like(eligible)
    resolved_depth = torch.where(used, oriented_plane_depth, plane_depth)

    classification_rgb = torch.zeros(
        (3, *plane_depth.shape[-2:]),
        device=plane_depth.device,
        dtype=plane_depth.dtype,
    )

    def paint(mask: torch.Tensor, color: tuple[float, float, float]) -> None:
        color_tensor = torch.tensor(
            color,
            device=classification_rgb.device,
            dtype=classification_rgb.dtype,
        )[:, None, None]
        classification_rgb.copy_(torch.where(
            mask.repeat(3, 1, 1), color_tensor, classification_rgb
        ))

    paint(boundary, (1.0, 0.5, 0.0))
    paint(candidate_invalid, (1.0, 0.0, 1.0))
    paint(uncertain_reference, (0.0, 1.0, 1.0))
    paint(depth_disagreement, (1.0, 0.0, 0.0))
    paint(insufficient_front_margin, (1.0, 1.0, 0.0))
    paint(eligible, (0.0, 1.0, 0.0))

    return {
        "sand_orientation_recovery_candidate": invalid_plane_depth,
        "sand_orientation_recovery_boundary": boundary,
        "sand_orientation_recovery_candidate_invalid": candidate_invalid,
        "sand_orientation_recovery_uncertain_reference": uncertain_reference,
        "sand_orientation_recovery_depth_disagreement": depth_disagreement,
        "sand_orientation_recovery_insufficient_front_margin": (
            insufficient_front_margin
        ),
        "sand_orientation_recovery_eligible": eligible,
        "sand_orientation_recovery_used": used,
        "sand_orientation_recovery_classification": classification_rgb,
        "sand_orientation_recovery_interior": interior,
        "sand_orientation_recovery_candidate_depth_raw": oriented_plane_depth,
        "sand_orientation_recovery_center_residual_raw": (
            candidate_center_residual
        ),
        "sand_orientation_recovery_margin_raw": margin,
        "sand_orientation_recovery_resolved_depth": resolved_depth,
    }


def fallback_invalid_plane_depth(
    plane_depth: torch.Tensor,
    center_depth: torch.Tensor,
    accumulated_alpha: torch.Tensor,
    coverage: torch.Tensor,
    *,
    enabled: bool,
    alpha_min: float = 0.10,
    coverage_min: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conservatively replace only numerically invalid water plane depths.

    The plane rasterizer reconstructs depth by intersecting a camera ray with
    the alpha-composited tangent plane.  Near a singular denominator that depth
    can become zero, negative or non-finite even though fluid Gaussians cover
    the pixel.  Their alpha-weighted camera-space center depth is less exact but
    stable enough for a guarded fallback.

    This helper deliberately does *not* perform opaque-layer occlusion.  The
    caller must still compare the returned depth with source+sand depth, so a
    recovered candidate behind a table leg remains rejected.
    """

    if not (
        plane_depth.shape
        == center_depth.shape
        == accumulated_alpha.shape
        == coverage.shape
    ):
        raise ValueError("all water depth fallback buffers must have equal shapes")
    alpha_min = _finite_scalar(alpha_min, "alpha_min")
    coverage_min = _finite_scalar(coverage_min, "coverage_min")
    if not 0.0 <= alpha_min <= 1.0:
        raise ValueError("alpha_min must lie in [0,1]")
    if not 0.0 <= coverage_min <= 1.0:
        raise ValueError("coverage_min must lie in [0,1]")

    plane_valid = torch.isfinite(plane_depth) & (plane_depth > 0.0)
    center_valid = torch.isfinite(center_depth) & (center_depth > 0.0)
    fallback_used = (
        bool(enabled)
        & (~plane_valid)
        & center_valid
        & (accumulated_alpha >= alpha_min)
        & (coverage >= coverage_min)
    )
    resolved_depth = torch.where(fallback_used, center_depth, plane_depth)
    return resolved_depth, plane_valid, fallback_used


def opaque_surface_owner(
    coverage: torch.Tensor,
    in_front: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Choose the discrete G-buffer owner behind a continuous RGB edge.

    Depth and normals cannot be alpha-averaged without losing their geometric
    meaning.  This mask is intentionally separate from RGB coverage so P4 can
    march reflection/refraction rays against one well-defined surface.
    """

    threshold = _finite_scalar(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("owner threshold must lie in [0,1]")
    if coverage.shape != in_front.shape or coverage.shape != valid.shape:
        raise ValueError("coverage, in_front and valid must have equal shapes")
    return valid.to(torch.bool) & (coverage >= threshold) & (
        in_front >= threshold
    )


def camera_normals_to_world(
    normals_camera: torch.Tensor,
    world_view_rotation: torch.Tensor,
) -> torch.Tensor:
    """Transform a normalized ``[3,H,W]`` normal map back to world space.

    The camera transform uses row vectors and constructs camera normals as
    ``normal_world @ world_view_rotation``.  Applying the matrix inverse here
    exactly reverses that convention.  No camera-facing flip is performed: the
    caller must pass the stable physical surface orientation when that is what
    material shading requires.
    """

    if normals_camera.ndim != 3 or normals_camera.shape[0] != 3:
        raise ValueError("normals_camera must have shape [3,H,W]")
    if world_view_rotation.shape != (3, 3):
        raise ValueError("world_view_rotation must have shape [3,3]")
    inverse_rotation = torch.linalg.inv(world_view_rotation)
    normals_world = torch.matmul(
        normals_camera.movedim(0, -1), inverse_rotation
    )
    normals_world = F.normalize(normals_world, dim=-1, eps=1.0e-6)
    return normals_world.movedim(-1, 0)


def orient_plane_normals_toward_camera(
    normals_camera: torch.Tensor,
    centers_camera: torch.Tensor,
    *,
    orthographic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a camera-facing normal copy for candidate plane depth.

    The plane rasterizer reconstructs depth with ``d / (-n.r)`` and stores
    ``d=abs(n.center)``.  Because the absolute distance no longer changes sign
    with ``n``, a camera-facing copy avoids the sign mismatch before alpha
    blending.  Positive candidate depth alone is not visibility evidence, so
    callers must retain their stable production/shading normal and pass this
    copy through the conservative invalid-depth resolver.

    Returns the oriented normals and a boolean ``[N]`` flip mask.  A genuinely
    grazing normal is left unchanged and remains available to the normal
    invalid-depth guard instead of being hidden by an arbitrary epsilon flip.
    """

    if normals_camera.ndim != 2 or normals_camera.shape[-1] != 3:
        raise ValueError("normals_camera must have shape [N,3]")
    if centers_camera.shape != normals_camera.shape:
        raise ValueError("centers_camera must match normals_camera")
    if not torch.is_floating_point(normals_camera):
        raise ValueError("normals_camera must be floating point")
    if not torch.is_floating_point(centers_camera):
        raise ValueError("centers_camera must be floating point")

    # The CUDA perspective ray is [x/z, y/z, 1]. Visible Gaussian centers have
    # z>0, but the safe denominator keeps malformed/cullable inputs diagnostic
    # rather than introducing NaN while deciding their orientation.
    if orthographic:
        center_ray = torch.zeros_like(centers_camera)
        center_ray[:, 2] = 1.0
    else:
        center_z = centers_camera[:, 2]
        safe_z = torch.where(
            torch.isfinite(center_z) & (torch.abs(center_z) > 1.0e-8),
            center_z,
            torch.ones_like(center_z),
        )
        center_ray = centers_camera / safe_z[:, None]
    center_denominator = -torch.sum(normals_camera * center_ray, dim=1)
    flip_mask = torch.isfinite(center_denominator) & (center_denominator < 0.0)
    oriented = torch.where(
        flip_mask[:, None], -normals_camera, normals_camera
    )
    return oriented, flip_mask


def perspective_view_directions_camera(
    height: int,
    width: int,
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return per-pixel surface-to-camera directions as ``[3,H,W]``.

    Camera-space depth cancels during normalization, so the direction can be
    computed without another position rasterization.  This replaces P3's old
    constant ``[0,0,-1]`` approximation and gives P4 a perspective-correct view
    vector for Fresnel and BRDF evaluation.
    """

    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    fx_t = torch.as_tensor(fx, device=device, dtype=dtype)
    fy_t = torch.as_tensor(fy, device=device, dtype=dtype)
    cx_t = torch.as_tensor(cx, device=device, dtype=dtype)
    cy_t = torch.as_tensor(cy, device=device, dtype=dtype)
    if torch.any(fx_t <= 0.0) or torch.any(fy_t <= 0.0):
        raise ValueError("camera focal lengths must be positive")

    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    # The camera looks along +Z in the rasterized depth convention.
    # Therefore the direction from a visible surface back to the camera is
    # the negated pinhole ray.
    directions = torch.stack(
        (
            -(grid_u - cx_t) / fx_t,
            -(grid_v - cy_t) / fy_t,
            -torch.ones_like(grid_u),
        ),
        dim=0,
    )
    return F.normalize(directions, dim=0, eps=1.0e-6)


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    """Decode an sRGB tensor without changing its layout."""

    value = torch.clamp(value, 0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        torch.pow((value + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Encode a non-negative linear RGB tensor as sRGB."""

    value = torch.clamp_min(value, 0.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * torch.pow(value, 1.0 / 2.4) - 0.055,
    )


def split_sum_specular_reflectance(
    f0: torch.Tensor, fg_lookup: torch.Tensor
) -> torch.Tensor:
    """Combine dielectric F0 with a two-channel split-sum BRDF lookup.

    The preintegrated lookup stores the scale and bias terms conventionally
    named ``A`` and ``B``.  Keeping this small material operation independent
    from cubemap sampling makes its channel/shape contract CPU-testable while
    the environment light remains responsible for nvdiffrast texture access.
    """

    if f0.shape[-1] not in {1, 3}:
        raise ValueError("F0 must end in one or three channels")
    if fg_lookup.shape[-1] != 2:
        raise ValueError("split-sum lookup must end in two channels")
    if f0.shape[:-1] != fg_lookup.shape[:-1]:
        raise ValueError("F0 and split-sum lookup spatial shapes must match")
    if not torch.isfinite(f0).all() or not torch.isfinite(fg_lookup).all():
        raise ValueError("split-sum reflectance inputs must be finite")
    return torch.clamp(
        f0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2],
        min=0.0,
    )


def occluded_reflection_proxy(
    low_frequency_rgb: torch.Tensor,
    environment_rgb: torch.Tensor,
    *,
    darkness: float = 0.25,
    saturation: float = 0.45,
    environment_weight: float = 0.10,
) -> torch.Tensor:
    """Approximate reflection from an occluded surface absent from the G-buffer.

    A single-view 3DGS reconstruction may contain the visible top of an object
    but no reliable underside. Reusing that top color for a back-face SSR hit is
    geometrically wrong. This proxy intentionally retains only blurred local
    color context, reduces saturation and brightness, then adds a small rough
    environment term. Inputs and output are linear RGB with final dimension 3.
    """

    if low_frequency_rgb.shape != environment_rgb.shape:
        raise ValueError("proxy color and environment tensors must have equal shapes")
    if low_frequency_rgb.shape[-1] != 3:
        raise ValueError("proxy colors must store RGB in the final dimension")
    darkness = _finite_scalar(darkness, "darkness")
    saturation = _finite_scalar(saturation, "saturation")
    environment_weight = _finite_scalar(environment_weight, "environment_weight")
    if not 0.0 <= darkness <= 1.0:
        raise ValueError("darkness must lie in [0,1]")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation must lie in [0,1]")
    if not 0.0 <= environment_weight <= 1.0:
        raise ValueError("environment_weight must lie in [0,1]")

    luminance_weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        device=low_frequency_rgb.device,
        dtype=low_frequency_rgb.dtype,
    )
    gray = torch.sum(low_frequency_rgb * luminance_weights, dim=-1, keepdim=True)
    desaturated = gray + saturation * (low_frequency_rgb - gray)
    proxy = (
        (1.0 - environment_weight) * darkness * desaturated
        + environment_weight * environment_rgb
    )
    return torch.clamp(proxy, 0.0, 1.0)


def schlick_fresnel(
    cosine: torch.Tensor, f0: float | torch.Tensor = 0.02
) -> torch.Tensor:
    """Evaluate Schlick Fresnel for a dielectric water interface."""

    f0_tensor = torch.as_tensor(f0, device=cosine.device, dtype=cosine.dtype)
    if torch.any((f0_tensor < 0.0) | (f0_tensor > 1.0)):
        raise ValueError("F0 must lie in [0,1]")
    cosine = torch.clamp(cosine, 0.0, 1.0)
    return f0_tensor + (1.0 - f0_tensor) * torch.pow(1.0 - cosine, 5.0)


def beer_lambert_transmittance(
    path_length: torch.Tensor,
    absorption_rgb: torch.Tensor,
) -> torch.Tensor:
    """Return RGB transmission ``exp(-sigma_a * distance)`` in linear space."""

    if torch.any(path_length < 0.0) or torch.any(absorption_rgb < 0.0):
        raise ValueError("path length and absorption must be non-negative")
    return torch.exp(-path_length * absorption_rgb)


def screen_space_refraction_grid(
    normal_camera: torch.Tensor,
    water_thickness: torch.Tensor,
    pixel_scale: float,
) -> torch.Tensor:
    """Build a ``grid_sample`` grid for SandyGS's image-space refraction.

    The refraction grid scales the projected water normal by the rasterized
    free-water thickness. Dry or vanishingly thin cells approach the
    undistorted substrate continuously.
    """

    if normal_camera.ndim != 3 or normal_camera.shape[0] != 3:
        raise ValueError("normal_camera must have shape [3,H,W]")
    if water_thickness.shape != (1,) + tuple(normal_camera.shape[1:]):
        raise ValueError("water_thickness must have shape [1,H,W]")
    pixel_scale = _finite_scalar(pixel_scale, "pixel_scale")
    if pixel_scale < 0.0:
        raise ValueError("pixel_scale must be non-negative")
    _, height, width = normal_camera.shape
    x = torch.linspace(-1.0, 1.0, width, device=normal_camera.device,
                       dtype=normal_camera.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=normal_camera.device,
                       dtype=normal_camera.dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    offset_x = normal_camera[0] * water_thickness[0] * pixel_scale
    offset_y = normal_camera[1] * water_thickness[0] * pixel_scale
    if width > 1:
        offset_x = offset_x * (2.0 / (width - 1))
    if height > 1:
        offset_y = offset_y * (2.0 / (height - 1))
    return torch.stack((grid_x + offset_x, grid_y + offset_y), dim=-1)[None]


def normalize_environment_irradiance(
    irradiance: torch.Tensor,
    reference_rgb: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Normalize environment irradiance while preserving color/direction.

    The scene-owned LDR environment map is a reflection fallback, not
    a metrically calibrated light probe.  Dividing by its global luminance lets
    P3 use directional variation without inheriting an arbitrary exposure.
    Both inputs are expected in linear RGB with the channel in the last axis.
    """

    minimum = _finite_scalar(minimum, "minimum")
    maximum = _finite_scalar(maximum, "maximum")
    if not 0.0 <= minimum < maximum:
        raise ValueError("irradiance clamp must satisfy 0 <= min < max")
    if irradiance.shape[-1] != 3 or reference_rgb.shape[-1] != 3:
        raise ValueError("irradiance inputs must end in three RGB channels")
    weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        device=reference_rgb.device,
        dtype=reference_rgb.dtype,
    )
    reference_luminance = torch.sum(reference_rgb * weights, dim=-1).mean()
    normalized = irradiance / torch.clamp(reference_luminance, min=1.0e-6)
    return torch.clamp(normalized, minimum, maximum)


def validate_surface_gbuffer(
    coverage: torch.Tensor,
    depth: torch.Tensor,
    normal_world: torch.Tensor,
    valid: torch.Tensor,
    *,
    normal_tolerance: float = 2.0e-3,
) -> dict[str, float | int]:
    """Validate the opaque surface contract consumed by P4.

    Validation is intentionally performed on the actual rasterized tensors,
    rather than only on analytic surfels.  It catches NaNs, invalid positive
    depth and camera/world normal conversion errors before screen-space water
    rays make the failure harder to localize.
    """

    if coverage.shape != depth.shape or coverage.shape != valid.shape:
        raise ValueError("coverage, depth and valid must have shape [1,H,W]")
    if coverage.ndim != 3 or coverage.shape[0] != 1:
        raise ValueError("surface scalar buffers must have shape [1,H,W]")
    if normal_world.shape != (3, coverage.shape[1], coverage.shape[2]):
        raise ValueError("normal_world must have shape [3,H,W]")
    valid = valid.to(torch.bool)
    if not torch.any(valid):
        raise ValueError("surface G-buffer contains no valid pixels")
    if not torch.isfinite(coverage).all():
        raise ValueError("surface coverage contains NaN or infinity")
    if torch.any((coverage < 0.0) | (coverage > 1.0)):
        raise ValueError("surface coverage lies outside [0,1]")
    if not torch.isfinite(depth[valid]).all() or torch.any(depth[valid] <= 0.0):
        raise ValueError("valid surface depth must be finite and positive")
    valid_normal = valid.repeat(3, 1, 1)
    if not torch.isfinite(normal_world[valid_normal]).all():
        raise ValueError("valid surface normals contain NaN or infinity")
    lengths = torch.linalg.vector_norm(normal_world, dim=0, keepdim=True)[valid]
    maximum_normal_error = torch.max(torch.abs(lengths - 1.0))
    normal_tolerance = _finite_scalar(normal_tolerance, "normal_tolerance")
    if maximum_normal_error > normal_tolerance:
        raise ValueError(
            "world-space surface normals are not unit length: "
            f"max error {float(maximum_normal_error):.6g}"
        )
    return {
        "valid_pixels": int(valid.sum()),
        "coverage_min": float(coverage[valid].min()),
        "coverage_max": float(coverage[valid].max()),
        "depth_min": float(depth[valid].min()),
        "depth_max": float(depth[valid].max()),
        "normal_max_length_error": float(maximum_normal_error),
    }
