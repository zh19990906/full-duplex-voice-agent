#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${MODEL_CONFIG:-$REPOSITORY_ROOT/configs/models.yaml}"
MODEL_HOME="${MODEL_HOME:-$REPOSITORY_ROOT/models}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: model configuration not found: $CONFIG_FILE" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to inspect model configuration." >&2
  exit 1
fi
if ! command -v modelscope >/dev/null 2>&1; then
  echo "ERROR: ModelScope CLI not found (expected 'modelscope')." >&2
  exit 1
fi

mkdir -p "$MODEL_HOME"
echo "ModelScope deployment plan"
echo "  config: $CONFIG_FILE"
echo "  model home: $MODEL_HOME"
echo "  no models downloaded: configure approved offline artifacts before execution"
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  relative_path="${path#./models/}"
  echo "  target: $MODEL_HOME/$relative_path"
done < <(awk '/^[[:space:]]+path:[[:space:]]+\.\/models\// {print $2}' "$CONFIG_FILE")
