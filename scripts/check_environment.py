"""Validate the Python, CUDA, and native-extension environment for SandyGS."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def package_version(distributions: Iterable[str]) -> str:
    for name in distributions:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def import_required(label: str, module_name: str, distributions: Iterable[str]) -> None:
    importlib.import_module(module_name)
    print(f"[ok] {label}: {package_version(distributions)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile-jit",
        action="store_true",
        help="also compile and execute nvdiffrast/NVDIFFREC runtime CUDA plugins",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {major}.{minor}")

    required = [
        ("NumPy", "numpy", ("numpy",)),
        ("Pillow", "PIL", ("Pillow",)),
        ("OpenCV", "cv2", ("opencv-python",)),
        ("Open3D", "open3d", ("open3d",)),
        ("SciPy", "scipy", ("scipy",)),
        ("scikit-image", "skimage", ("scikit-image",)),
        ("scikit-learn", "sklearn", ("scikit-learn",)),
        ("PyYAML", "yaml", ("PyYAML",)),
        ("Kornia", "kornia", ("kornia",)),
        ("Trimesh", "trimesh", ("trimesh",)),
        ("PyTorch3D", "pytorch3d", ("pytorch3d",)),
        ("nvdiffrast", "nvdiffrast.torch", ("nvdiffrast",)),
        (
            "diff-plane-rasterization",
            "diff_plane_rasterization",
            ("diff_plane_rasterization", "diff-plane-rasterization"),
        ),
        ("simple-knn", "simple_knn._C", ("simple_knn", "simple-knn")),
        ("BVH tracing", "bvh_tracing._C", ("bvh_tracing", "bvh-tracing")),
    ]
    for label, module_name, distributions in required:
        import_required(label, module_name, distributions)

    importlib.import_module("lpipsPyTorch")
    print("[ok] repository LPIPS evaluator")

    # This import follows the real application import graph and catches missing
    # dependencies which isolated package imports may not expose.
    importlib.import_module("gaussian_renderer")
    print("[ok] SandyGS gaussian_renderer import")

    if args.compile_jit:
        import nvdiffrast.torch as dr

        texture = torch.ones((1, 2, 2, 3), dtype=torch.float32, device="cuda")
        uv = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device="cuda")
        sampled = dr.texture(texture, uv, filter_mode="linear")
        torch.cuda.synchronize()
        if sampled.shape != (1, 1, 1, 3):
            raise RuntimeError(f"unexpected nvdiffrast output shape: {sampled.shape}")
        print("[ok] nvdiffrast CUDA JIT")

        from scene.NVDIFFREC import renderutils

        points = torch.zeros((1, 1, 3), dtype=torch.float32, device="cuda")
        matrix = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0)
        transformed = renderutils.xfm_points(points, matrix)
        torch.cuda.synchronize()
        if transformed.shape != (1, 1, 4):
            raise RuntimeError(
                f"unexpected NVDIFFREC transform output shape: {transformed.shape}"
            )
        print("[ok] NVDIFFREC renderutils CUDA JIT")

    print("SandyGS environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
