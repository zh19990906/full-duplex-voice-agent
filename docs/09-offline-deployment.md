# Offline Deployment Design

## Goal

Support local deployment with replaceable model paths.

## Directory

Default:

```
./models
```

Example:

```
models/
  x2-turn/
  llm/
  tts/
  translation/
```

## Configuration

All model locations must be configurable.

Example:

```yaml
models:
  turn:
    path: ./models/x2-turn
  llm:
    path: ./models/llm
```

## Download Sources

Supported:

- HuggingFace
- ModelScope

Scripts:

```
scripts/download_models_hf.sh
scripts/download_models_modelscope.sh
```

## Deployment Principle

Do not hard-code model locations.

The same runtime should support:

- development machine
- offline server
- edge device
