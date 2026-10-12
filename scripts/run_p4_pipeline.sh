#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sandygs" ]]; then
  echo "Activate the sandygs conda environment before running P4." >&2
  exit 2
fi

# P4 shares P3's immutable solver/render driver but selects a separate artifact
# namespace and enables the layered-water contract checks.
source "$(dirname "$0")/activate_environment.sh"
exec python "$(dirname "$0")/run_p3_pipeline.py" --stage p4 "$@"
