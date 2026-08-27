# Task 13 Report

## Scope Completed

- Restored `src/runtime_app/container.py` from `HEAD` behavior and extended it with realtime runtime composition instead of deleting it.
- Replaced the old serial real-server path with a realtime app bootstrap built around:
  - `RealtimeServerSettings`
  - `build_realtime_app(...)`
  - per-session `ServerRealtimeSessionRuntime`
- Preserved the interrupted `audio_ingress` and `session_runtime` additions:
  - `AudioIngress.push_decoded(...)`
  - `RealtimeSessionRuntime.accept_audio_frame(...)`
- Removed the direct `RealModelSession` / `model.run()` / global lock path from `scripts/run_real_server.py`.
- Added a production runtime factory that composes the real ASR / X2 / policy / Qwen / CosyVoice boundaries through existing realtime/runtime adapters.
- Kept websocket compatibility for both `/ws/{session_id}` and `/ws?session_id=...`.
- Added tolerant websocket message handling:
  - malformed JSON and malformed binary frames emit in-band `error` events
  - the connection stays alive for subsequent valid messages
- Preserved FastAPI lifespan cleanup and session delete/disconnect cleanup.
- Aligned the frontend message send path to websocket command submission instead of the old synchronous `/message` response path.
- Updated event serialization so response/runtime identity is available at the envelope level for playback/runtime flows.

## Review Fix Follow-up

- Reworked the production runtime factory so it caches only shared heavyweight loaded runtimes and creates fresh per-session ASR / turn / policy / chat-LLM / TTS / translation wrappers.
- Added injected runtime-loader seams for the checked-in real stack paths:
  - faster-whisper ASR
  - X2 turn detection
  - Qwen policy
  - Qwen-compatible chat/translation
  - CosyVoice worker runtime
- Added a production factory smoke test that uses injected loaded-runtime doubles but still follows the real composition shape and creates a session.
- Wired live interpretation mode in `ServerRealtimeSessionRuntime`:
  - committed revision-aware ASR chunks are routed into `accept_transcript_chunk(...)` only while interpretation mode is active
  - a runtime-backed translation sink now publishes translation events, TTS audio, playback identity, and checkpoint state
- Added bounded outbound event backpressure:
  - queue capacity is configurable
  - queue overflow raises an explicit slow-consumer failure
  - the runtime closes cleanly after preserving already-buffered events
  - websocket slow-consumer cleanup now closes the websocket, closes the runtime, and removes the dead session
- Threaded the real `audio_config` output sample rate/channel settings into runtime audio event publication instead of leaving that CLI/config path unused.
- Aligned the frontend to accept and display `translation` websocket events.

## Review Fix Follow-up Round 2

- Replaced the no-op shared-runtime proxy controls with an owner-aware boundary:
  - active `call(...)` and `iterate(...)` operations register the owning proxy
  - `cancel(...)`, `interrupt(...)`, and `reset(...)` are forwarded out-of-band only for the active owner or while the boundary is idle
  - a waiting session cannot cancel another session's active LLM/TTS operation
  - owner state is cleared in `finally` so later sessions can safely reset/use the shared runtime
- Preserved request-scoped cancellation arguments across the shared boundary for CosyVoice-style `request_id` cancellation.
- Replaced the zero-config default X2 loader with a profile-aware loader built on the real `X2TurnConfig` / `X2TurnAdapter` contract.
- The default X2 loader now honors resolved `model_path` / `local_path`, `device`, loader options, and CLI turn-model overrides.
- Added regressions covering:
  - prompt owner LLM cancel while a shared stream is blocked
  - prompt owner TTS cancel with preserved `request_id`
  - waiting-session cancel isolation
  - post-owner reset forwarding without deadlock
  - profile-aware default X2 loader arguments
  - bounded default turn profile shape

## Verification

Targeted Task 13 tests:

```bash
python3 -m unittest -q tests.test_runtime_app_container tests.test_realtime_bootstrap tests.test_realtime_websocket_flow tests.test_real_server_script tests.test_real_server_annotations tests.test_interpretation_runtime_integration tests.test_web_ui
```

Result: PASS

Frontend Node tests:

```bash
node --test tests/*.mjs
```

Result: PASS

Additional touched-area validation:

```bash
python3 -m unittest -q tests.test_runtime_app_container tests.test_realtime_bootstrap tests.test_realtime_websocket_flow tests.test_real_server_script tests.test_real_server_annotations tests.test_interpretation_runtime_integration tests.test_web_ui
```

Result: PASS

Full Python suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

Result: PASS (`Ran 489 tests ... OK`)

Frontend full suite:

```bash
node --test tests/*.mjs
```

Result: PASS (`33` tests)

## Notes

- The production runtime factory is lazy: building the FastAPI app does not eagerly load real models; the shared adapter bundle is created on first session creation.
- The flat `configs/models.yaml` shape was preserved for the existing resolver/deployment tests, and the production runtime factory adapts those profiles to the realtime-specific providers at composition time.
