#!/usr/bin/env bash
set -e

MODEL_ROOT=${MODEL_ROOT:-./models}
mkdir -p "$MODEL_ROOT"

# Configure models here.
# Example:
# modelscope download --model MODEL_NAME --local_dir "$MODEL_ROOT/model_name"

echo "Download ModelScope models into $MODEL_ROOT"
