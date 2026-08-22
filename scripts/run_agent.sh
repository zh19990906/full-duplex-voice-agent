#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required to run the voice agent" >&2
  exit 1
fi
if [[ ! -f "${REPOSITORY_ROOT}/configs/models.yaml" ]]; then
  echo "error: model configuration not found" >&2
  exit 1
fi

cd "${REPOSITORY_ROOT}"
exec python3 -m src.application.entrypoint "$@"
