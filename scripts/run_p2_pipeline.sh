#!/usr/bin/env bash
set -euo pipefail

# This is an executable wrapper, not a file to source into the interactive
# shell. Use $0 so an explicit `zsh scripts/run_p2_pipeline.sh ...` invocation
# resolves the same scripts directory as Bash without relying on BASH_SOURCE.
_sandygs_script_dir="$(cd "$(dirname "$0")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sandygs" ]]; then
    echo "Activate the sandygs Conda environment before running P2." >&2
    exit 1
fi

# CUDA 11.8 requires the GCC 11 toolchain recorded during P0. This also sets
# TORCH_CUDA_ARCH_LIST for the currently visible GPU before NVDIFFREC JIT runs.
source "${_sandygs_script_dir}/activate_environment.sh"
exec python "${_sandygs_script_dir}/run_p2_pipeline.py" "$@"
