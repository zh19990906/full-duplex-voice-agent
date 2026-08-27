# Task 14 Report

## Scope

- Added `WorkerSupervisor` with one-restart-max terminal failure reporting.
- Added reconnect snapshot publication from the production realtime runtime.
- Added explicit degradation handlers for ASR failure, policy timeout, playback failure, and X2 timeout fallback.
- Added VRAM safe-budget accounting and diagnostics in `src/model_runtime/manager.py`.
- Added acceptance-gate evaluation with explicit labels: `unit`, `simulated`, `model-integration`, `hardware-e2e`.
- Added `scripts/run_realtime_acceptance.py` and updated operator docs/environment checks.
- Wired new browser-visible runtime events into the checked-in frontend.

## Runtime resilience / VRAM fix slice

- Bounded worker supervision now performs at most one restart after backoff, propagates cancellation, and closes idempotently.
- Final-ASR failure aborts policy/generation; X2 timeout fallback requires silent activity plus stable/final ASR evidence.
- LLM failure pauses and preserves the response checkpoint and publishes identity-bearing `llm_failed`.
- Production TTS recovery closes and replaces the shared worker, then resubmits checkpoint-derived unsynthesized text once with epoch/cancellation fencing. Reload failure remains terminal TTS state, preserves and pauses the checkpoint, and publishes identity-bearing `tts_failed` and `worker_terminal` events without falling through to `llm_failed`.
- VRAM admission uses an explicit budget or CUDA device capacity and atomically accounts pending and loaded reservations. Committed loaded capacity retains `max(device-wide measured or conservative model bytes, requested bytes) + runtime overhead`; zero/unobservable deltas cannot erase a nonzero estimate.
- Factory shutdown attempts every resource close and releases each reservation in a per-resource `finally`, raising collected cleanup errors only after all resources have been attempted.
- The frontend event allowlist and status handling now include `llm_failed`.

Focused verification for this slice is recorded in the commit report; hardware/model-integration gates remain external.

Round-two runtime verification: `python3 -m unittest tests.test_model_memory_budget tests.test_runtime_app_container tests.test_realtime_degradation -v` — PASS (`39` tests).

## Verification

Executed locally:

```bash
python3 -m unittest tests.test_realtime_supervisor tests.test_realtime_snapshot tests.test_realtime_acceptance tests.test_realtime_degradation tests.test_model_memory_budget -v
python3 -m unittest tests.test_runtime_app_container tests.test_realtime_websocket_flow tests.test_real_server_script tests.test_real_hardware_benchmark tests.test_benchmarks_validation tests.test_tts_provider -v
python3 -m unittest discover -s tests -p 'test_*.py' -q
node --test tests/*.mjs
./scripts/run_demo.sh
python3 scripts/run_realtime_acceptance.py --help
./scripts/check_environment.sh
```

Results:

- Python unit and integration suite: PASS (`Ran 504 tests ... OK`)
- Frontend Node suite: PASS (`33` tests)
- Demo scenarios: PASS
- Acceptance CLI help: PASS
- Environment check: PARTIAL
  - Missing locally: `hf`/`huggingface-cli`
  - Missing locally: `modelscope`
  - Realtime acceptance CLI check: PASS

## Capability Matrix

- `unit`: PASS
- `simulated`: PASS
- `model-integration`: NOT RUN
- `hardware-e2e`: NOT RUN

The local run did not exercise real GPU-recorded model integration, real browser/headset E2E, or the 60-minute soak gate. No synthetic or partial result was labeled `hardware-e2e`.

## External Gates Not Run

Recorded-audio model integration on the GPU server:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label model-integration \
  --audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --report /tmp/realtime-model-integration.json
```

Real browser/headset E2E on the target machine:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label hardware-e2e \
  --explicit-real-run \
  --report /tmp/realtime-hardware-e2e.json
```

60-minute soak:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label hardware-e2e \
  --explicit-real-run \
  --report /tmp/realtime-soak.json
```

Required manual confirmation for the headset/soak gates:

- `duck_latency_ms <= 100`
- `interrupt_latency_ms <= 250`
- `backchannel_restore_latency_ms <= 300`
- `first_token_latency_ms <= 800`
- `first_audio_latency_ms <= 1500`
- `first_translated_audio_latency_ms <= 2000`
- `stale_output_count == 0`
- `resume_phrase_error_count <= 1`
- No unbounded queue growth
- No worker restart loop
- No CUDA OOM

## Conflicts

- A standalone `task-14-brief` file was not present anywhere under `/private/tmp/full-duplex-voice-agent-real-stack`; implementation used the checked-in Task 14 plan section, the design spec, and `task-13-report.md`.
