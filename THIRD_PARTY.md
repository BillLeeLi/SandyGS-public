# Third-party code and assets

SandyGS combines project-specific code with bundled and external components. [LICENSE.md](LICENSE.md) states the scope of the SandyGS PolyForm Noncommercial grant. Copyright notices in individual source files remain in place. The license copies below continue to apply to their respective components, including modified upstream code.

| Component | Location in this source tree | Source and notice |
|---|---|---|
| PGSR | Reconstruction and Gaussian scene code, including `arguments/`, parts of `gaussian_renderer/`, `scene/`, `utils/`, `train.py`, and `render.py`; this includes locally modified PGSR-based files | [PGSR](https://github.com/zju3dv/PGSR); full text: `THIRD_PARTY_LICENSES/PGSR-LICENSE.md` |
| 3D Gaussian Splatting and rasterization | `submodules/diff-plane-rasterization/`, `submodules/simple-knn/`, and inherited Gaussian infrastructure, including retained code in mixed rendering and camera files | [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); full text: `THIRD_PARTY_LICENSES/3DGS-LICENSE.md` and `submodules/diff-plane-rasterization/LICENSE.md` |
| GLM | `submodules/diff-plane-rasterization/third_party/glm/` | License text: `submodules/diff-plane-rasterization/third_party/glm/copying.txt` |
| NVDIFFREC utilities and LUT | `scene/NVDIFFREC/`, including local modifications | [nvdiffrec](https://github.com/NVlabs/nvdiffrec); full text: `THIRD_PARTY_LICENSES/NVDIFFREC-LICENSE.txt` |
| BVH | `bvh/` | Local license: `bvh/LICENCE` |
| Poly Haven HDR probe | `scene/NVDIFFREC/irrmaps/aerodynamics_workshop_2k.hdr` | [Asset page](https://polyhaven.com/a/aerodynamics_workshop), [CC0 terms](https://polyhaven.com/license), and adjacent `README.txt` |

PyTorch3D and nvdiffrast are installed from their pinned source revisions in `requirements.txt`. SAM, SAGA, SDXL, InFusion, and Grounding DINO are optional or stage-specific external model integrations; their source and pretrained weights are not included here. The photographed scene, trained checkpoints, sand textures, and rendered outputs are also external resources with separate terms.

The SandyGS renderer, camera scheduling, and reconstruction extensions are presented as parts of the SandyGS method. Some of their source files combine SandyGS contributions with PGSR or 3D Gaussian Splatting code; their upstream portions and notices retain the terms above. The project license covers only contributions that the SandyGS contributors can license. It does not override any third-party terms or make the whole repository available under one license.
