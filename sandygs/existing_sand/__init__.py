"""Offline provenance and artifact contracts for existing-sand experiments."""

from .workspace import (
    ExistingSandError,
    initialize_run,
    record_stage,
    verify_run,
)

from .geometry import (
    backproject_camera_z,
    crop_resize_intrinsics,
    depth_products,
    project_world,
    transform_world,
    validate_pinhole,
)

__all__ = [
    "ExistingSandError", "initialize_run", "record_stage", "verify_run",
    "backproject_camera_z", "crop_resize_intrinsics", "depth_products",
    "project_world", "transform_world", "validate_pinhole",
]
