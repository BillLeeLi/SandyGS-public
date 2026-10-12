"""E4 orthographic PGSR surface export, before any terrain filling or simulation."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .e3_checkpoint import fitted_checkpoint
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage, verify_run


def orthographic_grid(bounds: tuple[float, float, float, float], width: int, height: int):
    """Pixel [row,col] samples (x increasing right, y increasing down).

    PGSR's ndc2Pix maps NDC -1 to raster row zero, and its orthographic
    projection maps ymin to NDC -1 for this downward-facing camera.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmax <= xmin or ymax <= ymin:
        raise ExistingSandError("orthographic bounds must be finite xmin xmax ymin ymax")
    if width < 16 or height < 16 or width % 2 or height % 2:
        raise ExistingSandError("orthographic width/height must be even integers >= 16")
    dx, dy = (xmax - xmin) / width, (ymax - ymin) / height
    if not np.isclose(dx, dy, rtol=0, atol=1e-9):
        raise ExistingSandError("orthographic pixels must be square for the solver grid")
    x = xmin + (np.arange(width) + .5) * dx
    y = ymin + (np.arange(height) + .5) * dy
    return np.meshgrid(x, y, indexing="xy"), float(dx)


def image_yx_to_exchange_xy(array: np.ndarray) -> np.ndarray:
    """Map PGSR raster pixels to +x/+y cell centers, without interpolation."""
    if array.ndim != 2:
        raise ExistingSandError("orthographic field must be a 2D image")
    return np.ascontiguousarray(array.T)


def similarity_from_run(run_record: dict) -> tuple[np.ndarray, float]:
    matrix = np.asarray(run_record["coordinate"]["sim_to_gs"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ExistingSandError("run sim_to_gs must be a finite affine 4x4 matrix")
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    rotation = matrix[:3, :3] / scale
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-5, atol=1e-7):
        raise ExistingSandError("E4 orthographic export needs uniform sim-to-GS scale")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .9999:
        raise ExistingSandError("E4 orthographic export needs a proper similarity transform")
    return matrix, scale


def valid_height(depth: np.ndarray, alpha: np.ndarray, camera_z: float, scale: float,
                 alpha_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.shape != alpha.shape or depth.ndim != 2:
        raise ExistingSandError("depth and alpha must be matching 2D images")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= alpha_min)
    clean_depth = np.where(valid, depth, np.nan).astype(np.float32)
    height = np.where(valid, camera_z - depth / scale, np.nan).astype(np.float32)
    return clean_depth, height, valid


def surface_comparison(sand_height: np.ndarray, sand_valid: np.ndarray,
                       base_height: np.ndarray, base_valid: np.ndarray):
    if (sand_height.shape != base_height.shape or sand_valid.shape != sand_height.shape
            or base_valid.shape != base_height.shape):
        raise ExistingSandError("sand and background orthographic grids differ")
    both = sand_valid & base_valid & np.isfinite(sand_height) & np.isfinite(base_height)
    signed = np.full(sand_height.shape, np.nan, dtype=np.float32)
    signed[both] = sand_height[both] - base_height[both]
    return signed, both


def _quantiles(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(v) for v in np.quantile(finite, [.01, .5, .99])]


def _height_png(height: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> Image.Image:
    t = np.clip((np.nan_to_num(height, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack((255 * t, 255 * (1 - abs(2 * t - 1)), 255 * (1 - t)), axis=-1)
    rgb[~valid] = [25, 25, 25]
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def _difference_png(delta: np.ndarray, valid: np.ndarray) -> Image.Image:
    finite = np.abs(delta[valid])
    lim = max(float(np.quantile(finite, .99)) if finite.size else 0.0, .02)
    t = np.clip(np.nan_to_num(delta) / lim, -1, 1)
    rgb = np.empty(delta.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 128 + 127 * np.maximum(t, 0)
    rgb[..., 1] = 128 - 105 * np.abs(t)
    rgb[..., 2] = 128 + 127 * np.maximum(-t, 0)
    rgb[~valid] = [25, 25, 25]
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def _render_cloud(cloud: Path, sh_degree: int, white_background: bool,
                  c2w_gs: np.ndarray, bounds_gs: tuple[float, float, float, float],
                  width: int, height: int):
    import torch
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(sh_degree)
    model.load_ply(str(cloud))
    w2c = np.linalg.inv(c2w_gs)
    # Camera is the PGSR camera container. The orthographic pose is installed
    # explicitly because its x/y/z orientation includes a downward view.
    view = Camera(colmap_id=0, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                  FoVx=np.pi / 2, FoVy=np.pi / 2,
                  image_width=width, image_height=height, image_path="",
                  image_name="e4_orthographic", uid=0, preload_img=False)
    view.world_view_transform = torch.as_tensor(w2c.T.copy(), dtype=torch.float32, device="cuda")
    view.camera_center = torch.as_tensor(c2w_gs[:3, 3], dtype=torch.float32, device="cuda")
    view.c2w = torch.as_tensor(c2w_gs, dtype=torch.float32, device="cuda")
    args = SimpleNamespace(height_map_range=bounds_gs, height_map_width=width // 2,
                           height_map_height=height // 2, dataset_name="indoor02")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.ones(3, device="cuda") if white_background else torch.zeros(3, device="cuda")
    with torch.no_grad():
        out = render(view, model, pipe, bg, return_depth_normal=False,
                     is_training=False, is_render_height_map=True, args=args)
    rgb = out["render"].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    depth = out["plane_depth"].detach().cpu().numpy().squeeze().astype(np.float32)
    alpha = out["rendered_alpha"].detach().cpu().numpy().squeeze().astype(np.float32)
    if rgb.shape != (height, width, 3) or depth.shape != (height, width) or alpha.shape != depth.shape:
        raise ExistingSandError("orthographic renderer returned unexpected shape")
    del out, model, view
    torch.cuda.empty_cache()
    return rgb, depth, alpha


def _save_surface(directory: Path, name: str, rgb: np.ndarray, depth: np.ndarray,
                  alpha: np.ndarray, camera_z: float, scale: float, alpha_min: float):
    depth, height, valid = valid_height(depth, alpha, camera_z, scale, alpha_min)
    np.save(directory / f"{name}_depth_camera_z.npy", depth, allow_pickle=False)
    np.save(directory / f"{name}_alpha.npy", alpha, allow_pickle=False)
    np.save(directory / f"{name}_surface_height_yx.npy", height, allow_pickle=False)
    np.save(directory / f"{name}_valid_yx.npy", valid, allow_pickle=False)
    Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(directory / f"{name}_rgb.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(directory / f"{name}_valid.png")
    return height, valid


def export_orthographic_surfaces(run_root: Path, derived_id: str, fit_id: str, stage_id: str,
                                 bounds: tuple[float, float, float, float], width: int, height: int,
                                 camera_z: float, *, alpha_min: float, repo_root: Path,
                                 command: list[str] | None = None) -> dict:
    """Register untouched source, selected sand and completed background renders.

    This is an inspection stage. Invalid values remain NaN and signed thickness
    is not clipped, filled, or sent to the solver.
    """
    import torch

    run, repo = Path(run_root).resolve(), Path(repo_root).resolve()
    if not torch.cuda.is_available():
        raise ExistingSandError("E4 PGSR orthographic export requires CUDA")
    if not np.isfinite(camera_z) or not 0 < alpha_min < 1:
        raise ExistingSandError("camera_z must be finite and alpha_min in (0,1)")
    (_, _), spacing = orthographic_grid(bounds, width, height)
    target = _stage_location(run, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    verify_run(run)
    run_record = _load_yaml(run / "run.yaml", "existing-sand run")
    transform, scale = similarity_from_run(run_record)
    derived = _stage_location(run, "background", derived_id)
    if not (derived / "manifest.yaml").is_file():
        raise ExistingSandError("E4 needs a registered derived E3 scene")
    derived_rec = _load_yaml(derived / "record.yaml", "E3 derived record")
    if derived_rec.get("prompt", {}).get("type") != "approved_gaussian_deletion":
        raise ExistingSandError("E4 derived stage is not approved sand deletion")
    fit, fit_cloud_rel, fit_cloud, _ = fitted_checkpoint(run, fit_id, derived_id)
    selected_cloud = derived / "selected_sand.ply"
    if not selected_cloud.is_file() or "selected_sand.ply" not in {
        item["path"] for item in _load_yaml(derived / "manifest.yaml", "derived manifest")["artifacts"]
    }:
        raise ExistingSandError("registered selected_sand.ply is required")
    capture = [item["stage_id"] for item in derived_rec["parents"] if item["stage"] == "observations"]
    if len(capture) != 1:
        raise ExistingSandError("E3 deletion needs one observation source")
    obs = _stage_location(run, "observations", capture[0])
    obs_rec = _load_yaml(obs / "record.yaml", "PGSR observations")
    params = obs_rec["parameters"]
    sh_degree = params["sh_degree"]
    source_cloud = Path(run_record["source"]["checkpoint"]) / run_record["source"]["point_cloud"]
    if not source_cloud.is_file():
        raise ExistingSandError("original PGSR point cloud is missing")
    rotation = transform[:3, :3] / scale
    c2w = np.eye(4)
    c2w[:3, :3] = rotation @ np.diag([1.0, 1.0, -1.0])
    c2w[:3, 3] = transform[:3, :3] @ np.array([0.0, 0.0, camera_z]) + transform[:3, 3]
    xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    bounds_gs = tuple(scale * v for v in (xmin, xmax, ymin, ymax))
    payload = Path(tempfile.mkdtemp(prefix=f".e4_ortho_{stage_id}_", dir=run))
    try:
        outputs = {}
        for label, cloud in (("source", source_cloud), ("sand", selected_cloud), ("base", fit_cloud)):
            rgb, depth, alpha = _render_cloud(cloud, sh_degree, bool(params.get("white_background", False)),
                                              c2w, bounds_gs, width, height)
            outputs[label] = _save_surface(payload, label, rgb, depth, alpha, camera_z, scale, alpha_min)
        sand_h, sand_valid = outputs["sand"]
        base_h, base_valid = outputs["base"]
        source_h, source_valid = outputs["source"]
        delta, both = surface_comparison(sand_h, sand_valid, base_h, base_valid)
        if not np.any(both):
            raise ExistingSandError("selected sand and completed background have no shared valid pixels")
        np.save(payload / "sand_surface_xy.npy", image_yx_to_exchange_xy(sand_h), allow_pickle=False)
        np.save(payload / "base_surface_xy.npy", image_yx_to_exchange_xy(base_h), allow_pickle=False)
        np.save(payload / "raw_thickness_xy.npy", image_yx_to_exchange_xy(delta), allow_pickle=False)
        np.save(payload / "joint_valid_xy.npy", image_yx_to_exchange_xy(both), allow_pickle=False)
        source_overlap = source_valid & sand_valid
        source_delta = source_h[source_overlap] - sand_h[source_overlap]
        negative = both & (delta < 0)
        quality = {
            "resolution_xy": [width, height], "cell_spacing": spacing,
            "alpha_min": alpha_min, "source_valid_pixels": int(source_valid.sum()),
            "sand_valid_pixels": int(sand_valid.sum()), "base_valid_pixels": int(base_valid.sum()),
            "joint_valid_pixels": int(both.sum()), "sand_pixels_missing_base": int((sand_valid & ~base_valid).sum()),
            "negative_thickness_pixels": int(negative.sum()),
            "negative_thickness_fraction_of_joint": float(negative.sum() / both.sum()),
            "raw_thickness_p01_p50_p99": _quantiles(delta[both]),
            "negative_thickness_p01_p50_p99": _quantiles(delta[negative]),
            "source_vs_selected_height_p01_p50_p99": _quantiles(source_delta),
            "base_height_on_overlap_p01_p50_p99": _quantiles(base_h[both]),
            "sand_height_on_overlap_p01_p50_p99": _quantiles(sand_h[both]),
            "reference": "PGSR orthographic selected-sand and fitted-background surfaces; no hidden-surface truth",
        }
        with (payload / "quality.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(quality, stream, sort_keys=False, allow_unicode=True)
        finite = np.concatenate((sand_h[sand_valid], base_h[base_valid]))
        lo, hi = np.quantile(finite, [.01, .99])
        tiles = [
            ("source RGB", Image.open(payload / "source_rgb.png")),
            ("selected sand RGB", Image.open(payload / "sand_rgb.png")),
            ("base RGB", Image.open(payload / "base_rgb.png")),
            ("selected sand height", _height_png(sand_h, sand_valid, float(lo), float(hi))),
            ("base height", _height_png(base_h, base_valid, float(lo), float(hi))),
            ("sand minus base; red positive", _difference_png(delta, both)),
        ]
        overview = Image.new("RGB", (3 * width, 2 * (height + 22)), (255, 255, 255))
        draw = ImageDraw.Draw(overview)
        for index, (label, tile) in enumerate(tiles):
            x, y = (index % 3) * width, (index // 3) * (height + 22)
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
            overview.paste(tile, (x, y + 22))
        overview.save(payload / "overview.png")
        camera = {
            "type": "orthographic", "projection": "PGSR plane_depth camera-Z",
            "height_axis": "+simulation_z", "camera_z_simulation": float(camera_z),
            "image_size_wh": [width, height],
            "bounds_simulation_xy": [xmin, xmax, ymin, ymax],
            "cell_spacing_simulation": spacing,
            "pixel_to_simulation": "x=xmin+(column+0.5)*dx; y=ymin+(row+0.5)*dy",
            "exchange_conversion": "transpose image rows/columns to [x,y]",
            "c2w_gs": c2w.tolist(), "sim_to_gs": transform.tolist(),
            "gs_units_per_simulation_unit": scale,
        }
        with (payload / "camera.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(camera, stream, sort_keys=False, allow_unicode=True)
        record = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "same_orthographic_sand_background_review"},
            "seed": None, "models": [],
            "parameters": {"derived_id": derived_id, "fit_id": fit_id, "capture_id": capture[0],
                           "source_point_cloud": run_record["source"]["point_cloud"],
                           "selected_sand_point_cloud": "selected_sand.ply",
                           "fitted_point_cloud": f"checkpoint/{fit_cloud_rel}",
                           "bounds_simulation_xy": [xmin, xmax, ymin, ymax],
                           "resolution_xy": [width, height], "camera_z_simulation": float(camera_z),
                           "alpha_min": alpha_min, "height_processing": "none; invalid=NaN; signed thickness unclipped"},
            "commands": [command] if command else [], "command_cwd": str(repo),
            "parents": [{"stage": "background", "stage_id": derived_id},
                        {"stage": "background", "stage_id": fit_id},
                        {"stage": "observations", "stage_id": capture[0]}],
            "artifacts": [{"path": p.name, "role": "e4_orthographic_review"}
                          for p in sorted(payload.iterdir()) if p.is_file()],
            "code_files": ["sandygs/existing_sand/e4_height.py",
                           "scripts/experiments/existing_sand_e4.py"],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.rename(target)
        try:
            record_stage(run, target / "record.yaml", repo_root=repo)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {"stage_dir": str(target), **quality}
    finally:
        if payload.exists():
            shutil.rmtree(payload)
