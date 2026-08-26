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

## Verification

Targeted Task 13 tests:

```bash
python3 -m unittest tests.test_realtime_bootstrap tests.test_real_server_script tests.test_realtime_websocket_flow -v
```

Result: PASS

Frontend Node tests:

```bash
node --test tests/frontend_audio_capture_test.mjs tests/frontend_playback_test.mjs tests/frontend_streaming_test.mjs
```

Result: PASS

Frontend Python wrappers:

```bash
python3 -m unittest tests.test_frontend_audio_capture tests.test_frontend_streaming tests.test_frontend_playback tests.test_web_ui -v
```

Result: PASS

Additional touched-area validation:

```bash
python3 -m unittest tests.test_application_bootstrap tests.test_application_imports tests.test_real_server_annotations tests.test_web_ui tests.test_provider_factory tests.test_deployment_config tests.test_model_deployment tests.test_realtime_audio_ingress tests.test_real_backchannel_resume tests.test_interpretation_mode_policy tests.test_interpretation_runtime_integration -v
```

Result: PASS

Full Python suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

Result: PASS (`Ran 477 tests ... OK`)

Whitespace/sanity check:

```bash
git diff --check
```

Result: clean

## Notes

- The production runtime factory is lazy: building the FastAPI app does not eagerly load real models; the shared adapter bundle is created on first session creation.
- The flat `configs/models.yaml` shape was preserved for the existing resolver/deployment tests, and the production runtime factory adapts those profiles to the realtime-specific providers at composition time.
