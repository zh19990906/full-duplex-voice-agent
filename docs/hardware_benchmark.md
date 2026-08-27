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

Run or evaluate evidence against the Task 14 hard gates with an explicit label:

```bash
python3 scripts/run_realtime_acceptance.py --help
```

Valid labels are `unit`, `simulated`, `model-integration`, and
`hardware-e2e`. Imported metrics JSON is evaluable only as `unit` or
`simulated`; embedded provenance is ignored. Missing hard-gate metrics always
fail instead of being omitted from the decision.

The runner accepts a programmatic real-session hook that drives the existing
audio/model runtime and records events on `BenchmarkTimeline`. Record these
events to obtain latency metrics:

```text
audio_received -> first_asr_partial
turn_end -> first_llm_token
turn_end -> first_audio_chunk (first playable audio)
user_interrupt -> tts_stopped
cancel_requested -> generation_cancelled
```

Without a session hook, the runner reports only measured startup and process
memory values; missing interaction metrics are omitted rather than fabricated.

For recorded-audio model integration on the GPU server:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label model-integration \
  --audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --report /tmp/realtime-model-integration.json
```

The WAV must be PCM16 mono at 16 kHz. `--audio` drives the Task 13 production
realtime/session path in paced 20 ms frames and captures its timeline; it is
not copied into the report as an operator assertion. The runner derives
latencies from captured events and issues `recorded-audio-realtime`
provenance.

For imported simulated metrics instead:

```bash
python3 scripts/run_realtime_acceptance.py \
  --label simulated \
  --metrics-json /tmp/realtime-simulated-metrics.json \
  --report /tmp/realtime-simulated-acceptance.json
```

For real browser/headset validation:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label hardware-e2e \
  --browser-headset-driver /opt/voice-agent/bin/capture-browser-headset \
  --report /tmp/realtime-hardware-e2e.json
```

The executable receives `--profile <profile>` and must perform the physical
browser/headset run before printing a JSON object with captured `timeline`,
non-latency `metrics`, and `environment`. The acceptance runner stamps
`browser-headset` provenance only after that executable returns successfully;
there is no flag-only hardware attestation. The report is written to a sibling
temporary file and atomically replaced after the JSON is complete.

For the external 60-minute soak gate, keep the runtime on the real deployment
for one hour and confirm there is no unbounded queue growth, restart loop, CUDA
OOM, or stale audio. This gate is never inferred from synthetic or short runs.

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
