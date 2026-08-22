# Production Deployment

The production process is a thin deployment wrapper around the existing
application container. It loads a configuration profile, registers model
resources with `ModelLifecycleManager`, starts the application subscriptions,
and waits for `SIGINT` or `SIGTERM`. Provider resources are injected by the
deployment composition; this entrypoint does not construct providers or load
model SDKs itself.

## Requirements

- Python 3.10 or newer
- Configured provider dependencies for real model execution
- Prepared local model directories
- Audio device/backend dependencies when using hardware

The operations layer uses only the Python standard library. It does not
download models or install dependencies.

## Configuration

Profiles are selected with `VOICE_AGENT_PROFILE`:

```bash
VOICE_AGENT_PROFILE=dev ./scripts/run_agent.sh
VOICE_AGENT_PROFILE=local_gpu ./scripts/run_agent.sh
VOICE_AGENT_PROFILE=production ./scripts/run_agent.sh
```

Profiles live in `configs/profiles/`. Model locations remain relative and can
be relocated with the existing `MODEL_HOME` deployment setting.

## Startup

```bash
VOICE_AGENT_PROFILE=production ./scripts/run_agent.sh
```

When provider resources are injected, the entrypoint loads and warms profiles
marked `auto_load: true` before accepting application events. A config-only
startup registers model profiles but reports them as not ready until provider
resources are supplied. It emits JSON logs for startup, model readiness, and
shutdown.

## Shutdown

Send `SIGTERM` or press `Ctrl-C`. The process stops accepting new application
events, cancels the active generation, invokes the optional TTS stop boundary,
shuts down the application runtime, and unloads active model resources.

## Health and Status

`ProductionRuntime.health()` returns a JSON-compatible health snapshot. Its
status is `healthy`, `degraded`, or `unhealthy` based on application readiness
and model lifecycle states. `ProductionRuntime.status()` returns the active
profile, runtime state, model states, and monotonic uptime.

## Troubleshooting

### Model is missing

Prepare the configured relative model directory under `MODEL_HOME`, then
verify the corresponding `local_path` in `configs/models.yaml`.

### Audio device is unavailable

Check the selected audio provider in the profile and validate the host audio
backend separately. No hardware IDs are committed to this repository.

### Provider initialization fails

Inspect the structured `model_ready`/startup error logs, confirm provider
dependencies are installed, and verify the model path and device settings.

### Process exits immediately

Run the module directly to see configuration errors:

```bash
VOICE_AGENT_PROFILE=production python3 -m src.application.entrypoint
```
