#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_HOME="${MODEL_HOME:-$REPOSITORY_ROOT/models}"
status=0

check_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "OK: $command_name"
  else
    echo "MISSING: $command_name"
    status=1
  fi
}

echo "Environment check"
check_command python3
check_command git
if mkdir -p "$MODEL_HOME" && [[ -w "$MODEL_HOME" ]]; then
  echo "OK: writable model directory ($MODEL_HOME)"
else
  echo "MISSING: writable model directory ($MODEL_HOME)"
  status=1
fi
if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
  echo "OK: Hugging Face download tool"
else
  echo "MISSING: Hugging Face download tool (hf or huggingface-cli)"
  status=1
fi
if command -v modelscope >/dev/null 2>&1; then
  echo "OK: ModelScope download tool"
else
  echo "MISSING: ModelScope download tool (modelscope)"
  status=1
fi
exit "$status"
