#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${MODEL_CONFIG:-${REPOSITORY_ROOT}/configs/models.yaml}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required to resolve model profiles" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "error: model config not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ "${OFFLINE_MODE:-0}" != "1" ]] && ! command -v modelscope >/dev/null 2>&1; then
  echo "error: modelscope CLI is required; use OFFLINE_MODE=1 for directory preparation only" >&2
  exit 1
elif [[ "${OFFLINE_MODE:-0}" == "1" ]]; then
  echo "offline mode: skipping ModelScope CLI check"
fi

cd "${REPOSITORY_ROOT}"
args=(--config "${CONFIG_FILE}" --provider modelscope --prepare)
if [[ -n "${MODEL_HOME:-}" ]]; then
  args+=(--model-home "${MODEL_HOME}")
fi
python3 -m src.model_runtime.resolver "${args[@]}"
echo "No model was downloaded. Review the preview commands above before performing an approved download."
