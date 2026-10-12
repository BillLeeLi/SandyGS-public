# SandyGS

SandyGS combines a PGSR reconstruction with height-field sand–water simulation. It can select a sand pile in a reconstructed scene, remove its Gaussians, complete the exposed background, convert the recovered sand thickness to a solver initial condition, and render the evolving sand and water with the scene from multiple cameras.

This repository contains source code and example configurations. It does not include photographed datasets, trained reconstruction checkpoints, model weights, sand texture maps, solver binaries, simulation snapshots, or rendered videos.

## Pipeline

| Stage | Main entry points | Output |
|---|---|---|
| Reconstruct and inspect a scene | `train.py`, `render.py` | PGSR checkpoint and source views |
| Capture existing sand | `scripts/experiments/existing_sand.py`, `existing_sand_capture.py` | RGB, camera-Z depth, alpha, and cameras |
| Select sand | `scripts/experiments/existing_sand_e2.py` | SAM masks, multiview Gaussian labels, held-out checks |
| Complete the background | `scripts/experiments/existing_sand_e3.py`, `existing_sand_e3_multiview.py` | Derived checkpoint and fitted background |
| Build a height field | `scripts/experiments/existing_sand_e4.py` and the E4 repair/initialization entries | Terrain, sand thickness, and solver exchange |
| Simulate and render | `existing_sand_e4_pilot.py`, `existing_sand_e4_render.py`, `existing_sand_e4_demo.py` | Solver frames, RGB frames, and MP4 videos |

The E1 scripts provide an artificial-hole evaluation of RGB and depth completion. Optional E2 paths use SAGA-style affinity features or Grounding DINO text proposals; the core point/box and multiview selection path does not require them.

Each stage has a `--help` command. Stage IDs and source paths are explicit, and generated stage manifests record the inputs used for a run.

## Requirements

- Linux and an NVIDIA GPU with a CUDA toolkit compatible with the selected PyTorch build for PGSR training and rendering.
- Conda, Python 3.9, a C++20 compiler, and Xmake for the separate sand–water solver.
- COLMAP for preparing a new image collection and FFmpeg for MP4 encoding.
- External model code and weights for any enabled SAM, SAGA, SDXL, InFusion, or Grounding DINO stage.

The reference Python environment is specified in `environment.yml` and `requirements.txt`. The solver's C++ dependencies are declared in `solvers/sand_water/xmake.lua`.

Run the Xmake solver build as a non-root user from a checkout that user can write. Xmake downloads its declared C++ dependencies on the first build, so allow network access and set `TMPDIR` to a writable location if the system default points into a root-only directory. The solver build does not require CUDA.

`environment.yml` installs the PyTorch CUDA runtime, but does not install the CUDA compiler (`nvcc`). Before building the Python CUDA extensions, install a toolkit with `nvcc` and headers that matches `torch.version.cuda`, then set `CUDA_HOME` to that toolkit's root **before** sourcing `scripts/activate_environment.sh`. The script otherwise defaults to `/usr/local/cuda`, which may have a different version. Check the toolkit with `$CUDA_HOME/bin/nvcc --version`; the CUDA version shown by `nvidia-smi` describes driver support, not the compiler selected for the build.

```bash
conda env create -f environment.yml
conda activate sandygs
python -c 'import torch; print("PyTorch CUDA:", torch.version.cuda)'
export CUDA_HOME=/absolute/path/to/matching-cuda-toolkit
"$CUDA_HOME/bin/nvcc" --version
source scripts/activate_environment.sh
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation ./submodules/diff-plane-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./bvh
python scripts/check_environment.py
python scripts/build_sand_water_solver.py
python -m pytest tests
```

If the matching toolkit is installed inside the active Conda environment, use `export CUDA_HOME="$CONDA_PREFIX"` instead of the example path. This was the validated setup with PyTorch CUDA 11.8 and a Conda CUDA 11.8 toolkit. Confirm that `$CUDA_HOME/bin/nvcc` exists before installing the extensions.

Set the data and artifact roots on your machine, then create the ignored local configuration:

```bash
export SANDYGS_DATA_ROOT=/absolute/path/to/data
export SANDYGS_ARTIFACT_ROOT=/absolute/path/to/artifacts
test -e configs/paths.local.yaml || cp configs/paths.example.yaml configs/paths.local.yaml
```

Edit the local file if your dataset, checkpoints, or solver executable use different roots. Do not commit datasets, checkpoints, weights, local paths, or generated artifacts.

## External data and models

Keep downloaded resources and generated runs outside this repository. The paths below match `configs/paths.example.yaml` and the bundled `indoor02` configurations; choose your own absolute roots.

```text
$SANDYGS_DATA_ROOT/
├── datasets/<scene>/
│   ├── images/                                    # undistorted scene images
│   └── sparse/0/                                  # COLMAP camera reconstruction
├── checkpoints/pgsr/<scene>/                    # full trained PGSR checkpoint
│   ├── cfg_args
│   └── point_cloud/iteration_<N>/point_cloud.ply
├── checkpoints/existing-sand/
│   ├── sam/sam_vit_h_4b8939.pth
│   ├── infusion-depth/                          # complete InFusion Diffusers snapshot
│   └── sdxl-inpainting-0.1-fp16/                # complete SDXL fp16 Diffusers snapshot
└── materials/sand/texturecan/ground_0024_2k/runtime/
    ├── albedo.jpg
    ├── roughness.jpg
    └── normal_gl.png
$SANDYGS_CACHE_ROOT/src/segment-anything/
$SANDYGS_CACHE_ROOT/src/Infusion/
$SANDYGS_CACHE_ROOT/envs/existing-sand-inference/
$SANDYGS_ARTIFACT_ROOT/existing-sand/<scene>/<run-id>/
```

A PGSR checkpoint must contain its complete model directory and any auxiliary assets required by its training configuration; a PLY alone is insufficient. Use `train.py` with your own COLMAP-ready image collection, or supply a compatible checkpoint. The scene-to-simulation transform and physical scale must be chosen for that scene. The bundled PGSR and 3DGS code needs no separate clone.

| Stage | Obtain separately | Local check |
|---|---|---|
| E2 prompted masks | [Segment Anything source and ViT-H checkpoint](https://github.com/facebookresearch/segment-anything) | `segment-anything/segment_anything/__init__.py` and `sam_vit_h_4b8939.pth` |
| E3 RGB/depth completion | [InFusion source](https://github.com/ali-vilab/Infusion), its [depth checkpoint](https://huggingface.co/Johanan0528/Infusion/tree/main), and the [SDXL inpainting fp16 model](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | InFusion `depth_inpainting/run/run_inference_inpainting.py`; model directories containing `unet/`, `vae/`, and their configs/weights |
| Optional E2 affinity or text proposals | [SAGA source](https://github.com/Jumpat/SegAnyGAussians) or [Grounding DINO source and checkpoint](https://github.com/IDEA-Research/GroundingDINO) | Pass their code/model locations explicitly to the relevant E2 command |
| Textured final demo | [TextureCan Ground 0024](https://www.texturecan.com/details/215/) or equivalent compatible maps | All three `runtime/` maps above, at matching dimensions |

The frequency-vote E2 route does not require SAGA or Grounding DINO. SAGA-style affinity features, when used, are trained per scene in the PGSR-aligned E2 implementation; an upstream SAGA feature checkpoint is not a drop-in replacement. The optional Grounding DINO proposal branch has not been validated in the clean-checkout reproduction. Model weights and texture maps have their own distribution terms and are not included here. The SDXL and InFusion snapshots are multi-file directories, not single weight files; check available disk space before downloading them.

SDXL and InFusion use a separate Python environment layered on the main Conda environment. The `indoor02_multiview.yaml` example reads `SANDYGS_CACHE_ROOT`; for other scenes, pass these paths directly or edit a copy of that configuration.

```bash
export SANDYGS_CACHE_ROOT=/absolute/path/to/cache
mkdir -p "$SANDYGS_CACHE_ROOT/envs"
python -m venv --system-site-packages "$SANDYGS_CACHE_ROOT/envs/existing-sand-inference"
"$SANDYGS_CACHE_ROOT/envs/existing-sand-inference/bin/python" -m pip install \
  -r configs/requirements-existing-sand-inference.txt
```

Create the venv after activating `sandygs`, so it can reuse the installed PyTorch/CUDA stack. Keep the downloaded model directories intact: E3 checks their expected weight and config files before inference. Set `HF_HOME` to a cache directory outside the repository if the upstream tools need it.

## Reproducing with your own sand scene

This workflow has review points: the sand masks, deletion set, RGB/depth completions, fitted background, and height fields must be checked before feeding the next stage. Use new stage IDs when repeating a step; outputs are deliberately not overwritten. Commands below are run from the repository root. Camera names, pixel crops, bounds, and prompts are examples to replace with values from your scene.

**1. Register and capture a PGSR scene.** Write a private `run_input.yaml` outside Git, with the actual dataset and full checkpoint paths:

```yaml
schema_version: 1
scene_id: my_sand_scene
source:
  dataset: /absolute/path/to/data/datasets/my_sand_scene
  checkpoint: /absolute/path/to/data/checkpoints/pgsr/my_sand_scene
  point_cloud: point_cloud/iteration_30000/point_cloud.ply
  dataset_files:
    - sparse/0/cameras.bin
    - sparse/0/images.bin
    - images/frame_0001.jpg
    - images/frame_0002.jpg
    - images/frame_0003.jpg
camera_ids: [frame_0001.jpg, frame_0002.jpg, frame_0003.jpg]
coordinate:
  sim_to_gs:
    - [1, 0, 0, 0]
    - [0, 1, 0, 0]
    - [0, 0, 1, 0]
    - [0, 0, 0, 1]
```

The identity transform is only an example; set `sim_to_gs` to the measured scene alignment. List the actual images and camera files used for the run. `--dataset-name` must match the name used when training PGSR.

```bash
RUN="$SANDYGS_ARTIFACT_ROOT/existing-sand/my_sand_scene/run_001"
python scripts/check_artifact_storage.py
python scripts/experiments/existing_sand.py init \
  --config /absolute/path/to/run_input.yaml --output "$RUN" --run-id run_001
python scripts/experiments/existing_sand_capture.py \
  --run "$RUN" --capture-id source_views_001 --dataset-name my_sand_scene \
  --camera-id frame_0001.jpg --camera-id frame_0002.jpg \
  --camera-id frame_0003.jpg
```

Review `stages/observations/source_views_001/quality.yaml` and each `view_*/render_rgb.png`, `depth_valid.npy`, and `camera.yaml` before selecting sand.

**2. Prompt SAM, vote, and inspect held-out views.** Create a JSON file outside Git using full-resolution pixel coordinates. `positive` and `negative` are point lists; `box` is `[x_min, y_min, x_max, y_max]`. Holdout masks are references for checking and do not vote.

```json
{"views": [
  {"camera_id": "frame_0001.jpg", "role": "seed", "positive": [[520, 340]], "box": [410, 250, 680, 460]},
  {"camera_id": "frame_0002.jpg", "role": "seed", "positive": [[480, 350]], "negative": [[390, 350]]},
  {"camera_id": "frame_0003.jpg", "role": "holdout", "box": [300, 260, 660, 500]}
]}
```

```bash
python scripts/experiments/existing_sand_e2.py masks \
  --run "$RUN" --capture-id source_views_001 --mask-id e2_masks_001 \
  --prompts /absolute/path/to/sand_prompts.json \
  --sam-code "$SANDYGS_CACHE_ROOT/src/segment-anything" \
  --sam-checkpoint "$SANDYGS_DATA_ROOT/checkpoints/existing-sand/sam/sam_vit_h_4b8939.pth"
python scripts/experiments/existing_sand_e2.py select-frequency \
  --run "$RUN" --mask-id e2_masks_001 --stage-id e2_votes_001 \
  --sand-fraction 0.65
python scripts/experiments/existing_sand_e2.py check \
  --run "$RUN" --seed-id e2_votes_001 --check-id e2_check_001
```

`0.65` is an example vote threshold, not a universal sand setting. Review every SAM overlay and the held-out `stages/segmentation/e2_check_001/view_*/overlay.png` for missing sand and background leakage. The checker reports agreement with SAM masks, not independent ground-truth segmentation.

**3. Delete reviewed sand and fit the background.** The review file binds the exact E2 selection and source checkpoint. Inspect it, change `approved: false` to `approved: true` only after accepting the deletion set, then derive a new checkpoint copy:

```bash
python scripts/experiments/existing_sand_e3.py plan \
  --run "$RUN" --selection-id e2_votes_001 --check-id e2_check_001 \
  --output /absolute/path/to/e3_review.yaml
python scripts/experiments/existing_sand_e3.py derive \
  --run "$RUN" --selection-id e2_votes_001 --check-id e2_check_001 \
  --decision /absolute/path/to/e3_review.yaml --stage-id e3_derived_001
```

`prepare-background` renders the derived checkpoint and creates a shared RGB/depth edit mask. Choose a crop within the captured image; its longest side must be 768 pixels and both dimensions multiples of eight for the current InFusion adapter. Review `mask_overlay.png` and `quality.yaml` before inference. The example crop and tabletop prompt below must be adapted to the selected scene.

```bash
python scripts/experiments/existing_sand_e3.py prepare-background \
  --run "$RUN" --derived-id e3_derived_001 --camera-id frame_0001.jpg \
  --stage-id e3_edit_001 --crop 0 0 768 512
python scripts/experiments/existing_sand_e3.py run-rgb-depth \
  --run "$RUN" --prepare-id e3_edit_001 --stage-id e3_completion_001 \
  --seed 23 --prompt 'A continuous tabletop matching the surrounding surface.' \
  --model-python "$SANDYGS_CACHE_ROOT/envs/existing-sand-inference/bin/python" \
  --infusion-code "$SANDYGS_CACHE_ROOT/src/Infusion" \
  --infusion-model "$SANDYGS_DATA_ROOT/checkpoints/existing-sand/infusion-depth" \
  --sdxl-model "$SANDYGS_DATA_ROOT/checkpoints/existing-sand/sdxl-inpainting-0.1-fp16"
python scripts/experiments/existing_sand_e3.py initialize-3d \
  --run "$RUN" --completion-id e3_completion_001 --stage-id e3_init_001
python scripts/experiments/existing_sand_e3.py optimize-3d \
  --run "$RUN" --initialization-id e3_init_001 \
  --completion-ids e3_completion_001 --stage-id e3_fit_001
```

Inspect `sdxl_raw.png`, `completed_rgb.png`, `completed_depth.npy`, fitted RGB views, and held-out views. For a multiview fit, prepare and complete other training cameras, continue from the preceding fit with `--source-fit-id`, and pass the fixed completions to `optimize-3d --completion-ids ...` for alternating supervision. Keep holdout cameras out of that list. The bundled `indoor02_multiview.yaml` and `existing_sand_e3_multiview.py --step status/prepare/complete/initialize/fit/joint` show this scheduling pattern, but their IDs, crops, prompts, and resource paths are specific to `indoor02`.

**4. Review the height field and simulate.** Use `existing_sand_e4.py` to render the original sand top and fitted background with the *same orthographic camera*; `--bounds`, `--resolution`, and `--camera-z` must be chosen in your simulation coordinates. The retained E4 processing order is `existing_sand_e4_local_fit.py` → `existing_sand_e4_sand_surface.py` → `existing_sand_e4_edge_transition.py` → `existing_sand_e4_point_repair.py` → `existing_sand_e4_initialize.py`. Review each stage's overview/metrics and confirm a finite terrain and nonnegative sand thickness before solver initialization. Each entry exposes its required stage IDs through `--help`.

The current `existing_sand_e4_point_repair.py` and initializer specifically consume an E4 candidate with **exactly one reviewed internal invalid background pixel**. They encode the accepted `indoor02` repair and are not yet a general automatic repair for arbitrary scenes. A different scene or one with no such pixel needs an equivalent reviewed finite terrain/sand candidate before using the initializer. The `indoor02` initialization and pilot YAMLs are parameter examples; copy them and update the scene, geometry, water footprint, stage IDs, and duration rather than assuming their numerical values fit a new scene.

Finally, run `existing_sand.py verify --run "$RUN"` to check the registered input and stage hashes. Solver outputs live under `$RUN/simulations/<experiment-id>/water/`; rendered RGBs and videos live under `$RUN/visualizations/<render-id>/water/`.

## Re-rendering the bundled three-second demo

The final `indoor02` YAML records an accepted configuration, not downloadable demo inputs. To reproduce its exact frames, supply the same `indoor02` images/COLMAP data, trained PGSR checkpoint, completed offline run, and texture maps. The run must already contain observation stage `source_fourteen_views_001`, background stage `e4_position_full_fit300_001`, modeling stage `e4_exchange_initial_v2_annular_variant003_long300_001`, and water simulation `e4_water_pilot_v2_annular_variant003_long300_001`. These names are verified by `check`; changing only the path to an unrelated run cannot reproduce the result.

If you have the reviewed initialization stage but no solver output yet, run the solver once from the bundled pilot configuration:

```bash
RUN=/absolute/path/to/indoor02/run_002
PILOT=configs/existing_sand/indoor02_e4_pilot_v2_annular_variant003_long300.yaml
python scripts/experiments/existing_sand_e4_pilot.py \
  --run "$RUN" --config "$PILOT" --case water --check-only
python scripts/experiments/existing_sand_e4_pilot.py \
  --run "$RUN" --config "$PILOT" --case water
```

If that simulation already exists, skip both pilot commands. For a new render use a fresh `render_id` in a copy of the demo YAML; existing output directories are never overwritten. `render` reads solver snapshots and does not rerun the physics.

```bash
CFG=configs/existing_sand/indoor02_e4_demo_v2_annular_variant003_3s_24fps_texture.yaml
python scripts/experiments/existing_sand_e4_demo.py check --run "$RUN" --config "$CFG"
python scripts/experiments/existing_sand_e4_demo.py render --run "$RUN" --config "$CFG"
python scripts/experiments/existing_sand_e4_demo.py encode --run "$RUN" --config "$CFG"
```

`check` prints the resolved camera map, frame count, output path, and render command. The configured 3 s interval at 24 fps yields 73 frames per camera; detailed diagnostics are saved only at 0, 0.5, 1, 2, and 3 s. Inspect `visualizations/<render-id>/water/release_check/render_metrics.yaml`, a few `selected_views/time_*/renders/camera_*.png` frames, and `videos/manifest.yaml`; the latter lists the single-camera and combined MP4 files with hashes. The separate `indoor02_e4_render_*.yaml` is a sparse diagnostic configuration tied to another completed simulation, not a prerequisite for this final demo.

## License and third-party components

SandyGS-specific contributions are available under [PolyForm Noncommercial 1.0.0](LICENSE.md). This repository also contains PGSR, 3D Gaussian Splatting and rasterization code, NVDIFFREC utilities, GLM, BVH, and modified upstream files that retain their respective terms. See [THIRD_PARTY.md](THIRD_PARTY.md), the included license files, and original source notices before reusing code. External datasets, pretrained weights, and textures retain their own terms.
