# Offline model deployment

The default model root is `./models`. Model artifacts are intentionally not
committed to this repository.

## Directory structure

```text
models/
├── x2-turn/
├── asr/
├── llm/
├── tts/
└── translation/
```

## Offline workflow

1. Edit `configs/models.yaml` with the backend, model name, and local path for
   each model.
2. Set `MODEL_HOME` when the artifact root is outside the repository.
3. Run `scripts/check_environment.sh` to check Python, Git, directory access,
   and backend CLI availability.
4. Run the Hugging Face or ModelScope deployment script to validate the
   configuration and print its planned targets.
5. Place approved model artifacts into the configured directories through the
   organization's offline transfer process.

The deployment scripts are deliberately validation/planning placeholders for
this issue; they do not download or install anything.

## Configuration and replacement

`MODEL_CONFIG` can point deployment scripts at an alternate YAML file, while
`MODEL_HOME` overrides the default artifact root. To replace a model, update
its `backend`, `name`, and `path` in the configuration and place the new
artifact at that path. Runtime code can then use the same stable configured
location without a code change.
