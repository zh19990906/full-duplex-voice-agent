# Real Hardware Benchmark

The real hardware benchmark validates the deployment environment before
starting the existing application runtime. It does not substitute fake
devices or providers and it does not invent latency events.

## Requirements

- Python 3.10 or newer
- A physical microphone and speaker
- The `sounddevice` package and a working host audio backend
- Prepared ASR, LLM, and TTS model directories
- NVIDIA GPU and `nvidia-smi` when configured models use `device: cuda`
- Provider dependencies and runtime objects supplied by the deployment

## Configuration

Select the hardware profile with:

```bash
VOICE_AGENT_PROFILE=hardware_benchmark
```

The profile is stored in `configs/profiles/hardware_benchmark.yaml`. It uses
sounddevice for input/output and Whisper, llama.cpp, and CosyVoice provider
names without committing hardware IDs, absolute paths, or model files.

## Running

Run validation and the startup/system benchmark report:

```bash
VOICE_AGENT_PROFILE=hardware_benchmark \
  python3 -m benchmarks.real_hardware_runner
```

Use `--json` for machine-readable output:

```bash
python3 -m benchmarks.real_hardware_runner --json
```

The runner accepts a programmatic real-session hook that drives the existing
audio/model runtime and records events on `BenchmarkTimeline`. Record these
events to obtain latency metrics:

```text
audio_received -> first_asr_partial
turn_end -> first_llm_token
first_llm_token -> first_audio_chunk
user_interrupt -> tts_stopped
cancel_requested -> generation_cancelled
```

Without a session hook, the runner reports only measured startup and process
memory values; missing interaction metrics are omitted rather than fabricated.

## Validation failures

Validation fails before runtime startup when it detects:

- missing microphone
- missing speaker
- missing model directory
- unavailable GPU for a CUDA-configured model
- unavailable sounddevice or GPU probe

Failures are printed as an explicit list and never fall back to fake devices or
development providers.

## Troubleshooting

### Missing device

Verify the operating-system audio device is visible and that `sounddevice`
can enumerate it. The benchmark intentionally uses the default input/output
device and does not store machine-specific IDs.

### Missing model

Prepare the relative paths from `configs/models.yaml`, or set `MODEL_HOME` to
the directory containing the model folders. The benchmark checks directories
before startup.

### CUDA unavailable

Run `nvidia-smi`, verify the driver, and change the deployment environment only
after confirming the models and providers support the selected device. The
benchmark will not silently run a CUDA profile on CPU.

### Provider initialization failure

The benchmark runner does not construct provider SDK objects. Inject the real
provider resources through the deployment composition and inspect the
provider/runtime error before rerunning the benchmark.

