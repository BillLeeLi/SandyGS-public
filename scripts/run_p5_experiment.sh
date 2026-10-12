#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sandygs" ]]; then
  echo "Activate the sandygs Conda environment before running P5 experiments." >&2
  exit 2
fi

source "$(dirname "$0")/activate_environment.sh"
exec python "$(dirname "$0")/run_p3_pipeline.py" --stage p5 "$@"
