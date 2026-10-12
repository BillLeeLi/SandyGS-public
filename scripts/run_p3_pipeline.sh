#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sandygs" ]]; then
  echo "Activate the sandygs conda environment before running P3." >&2
  exit 2
fi

# Reuse the CUDA/GCC guard so native extensions use a compatible compiler.
source "$(dirname "$0")/activate_environment.sh"
exec python "$(dirname "$0")/run_p3_pipeline.py" "$@"
