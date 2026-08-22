# Offline model deployment

The default model root is `./models`. Model artifacts are intentionally not
committed to this repository.

## Directory structure

```text
models/
├── asr/
├── llm/
├── tts/
└── translation/
```

## Offline workflow

1. Edit `configs/models.yaml` with `provider`, `model_id`, `local_path`, and
   optional `device` for each profile.
2. Set `MODEL_HOME=/data/models` when the artifact root is outside the
   repository. The configured `./models/<profile>` paths then resolve beneath
   that directory.
3. Run `scripts/check_environment.sh` to check Python, Git, directory access,
   and provider CLI availability.
4. Run `OFFLINE_MODE=1 scripts/download_hf_models.sh` or
   `OFFLINE_MODE=1 scripts/download_modelscope_models.sh` to prepare directories
   and print provider commands without downloading anything.
5. Copy approved model artifacts through the organization's offline transfer
   process, then verify the resolved directories before starting a runtime.

The deployment scripts are deliberately validation/planning placeholders for
this issue; they do not download or install anything.

## Configuration and replacement

`MODEL_CONFIG` can point deployment scripts at an alternate YAML file, while
`MODEL_HOME` overrides the default artifact root. To replace a model, update
its `provider` and `model_id`, copy the replacement into the same `local_path`,
or change that path to a new relative directory. The runtime-facing resolver
keeps this deployment decision outside model adapters and business logic.
