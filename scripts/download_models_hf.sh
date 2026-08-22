#!/usr/bin/env bash
set -e

MODEL_ROOT=${MODEL_ROOT:-./models}
mkdir -p "$MODEL_ROOT"

# Configure models here.
# Example:
# huggingface-cli download MODEL_NAME --local-dir "$MODEL_ROOT/model_name"

echo "Download HuggingFace models into $MODEL_ROOT"
