# Task 3 Report: Browser PCM AudioWorklet Capture

## Delivered

- Added `frontend/src/capture-worklet.js` with a stateful linear resampler, exact 320-sample PCM16 framing, V1 big-endian header encoding, transferable `ArrayBuffer` posting, and a sequence counter that advances only for complete frames.
- Replaced MediaRecorder capture with `PcmMicrophoneInput`, which applies the required mono/AEC/noise suppression/AGC constraints and owns the stream, AudioContext, source, and worklet lifecycle. `MicrophoneInput` remains a compatibility alias.
- Updated the application to send the worklet's already-encoded `ArrayBuffer` directly through the WebSocket. Browser playback was not changed.
- Added Node behavior tests, including a real 320-sample frame, partial-frame withholding, PCM16 conversion, big-endian V1 header fields, stateful resampling, and repeated microphone start/stop lifecycle behavior.
- Added a Python cross-language contract test that invokes the JavaScript encoder and passes its unmodified binary frame to `decode_audio_frame`.

## TDD evidence

RED:

```text
$ node --test tests/frontend_audio_capture_test.mjs
Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../frontend/src/capture-worklet.js'
```

After a self-review found a block-boundary defect, a second focused RED test reproduced it:

```text
resampler does not skip a source sample when a 48kHz process block ends mid-phase
128 !== 129
```

GREEN:

```text
$ node --test tests/frontend_audio_capture_test.mjs
tests 4; pass 4; fail 0
```

## Verification commands and results

```text
$ node --test tests/frontend_audio_capture_test.mjs
4 passed, 0 failed

$ python3 -m unittest tests.test_frontend_audio_capture tests.test_web_ui tests.test_frontend_streaming -v
Ran 10 tests ... OK

$ python3 -m unittest discover -s tests
Ran 282 tests ... OK

$ git diff --check
exit 0; no output
```

`python -m unittest ...` could not run because this environment has no `python` executable; `python3` ran the identical unittest modules.

## Self-review

Reviewed every changed/new file against the V1 contract. The header is exactly 18 bytes (`!BIdIB` layout), payloads are exactly 640 bytes, PCM bytes are native little-endian `Int16Array` data, and the Python decoder accepts the JavaScript result unchanged. The resampler retains fractional phase across callbacks and no undersized frame is emitted. The implementation contains no MediaRecorder fallback and leaves `BrowserAudioOutput` unchanged. No subagents or external reviewers were used, per task instructions.

## Concerns

Automated tests use narrow browser fakes and Node cannot exercise a physical microphone or a browser's actual AudioWorklet scheduler. A manual modern-browser microphone check remains advisable before release; this does not affect the verified wire contract or lifecycle behavior.

## Fix Round 1

### Delivered

- Serialized Start-click replacement in `app.js`; an active microphone is stopped before the factory creates and starts its replacement.
- Resumed suspended capture contexts before connecting the microphone worklet.
- Made start-failure cleanup release locally allocated source/worklet/port as well as stream/context, and made release operations idempotent so one cleanup failure does not prevent the rest.
- Changed PCM payload serialization to explicit `DataView.setInt16(offset, sample, true)` writes and added literal byte assertions.

### TDD evidence

RED against `dc99fc6`:

```text
$ node --test tests/frontend_audio_capture_test.mjs
tests 8; pass 5; fail 3
- suspended context: expected ["resumed"], actual []
- setup failure cleanup: expected port/worklet/source cleanup, actual only track/context cleanup
- second Start click: attempted navigator.mediaDevices.getUserMedia instead of replacing through the microphone factory
```

The literal-byte test passed on the host before this fix because its native typed-array endianness is little-endian; the production implementation now explicitly writes the required byte order independently of that host property.

The failure-cleanup test was then strengthened to throw from `onStateChange("录音中")` after ownership assignment. Its RED showed all four instance fields still held the stream/context/source/worklet after local cleanup; GREEN clears all four fields before releasing those same resources.

GREEN:

```text
$ node --test tests/frontend_audio_capture_test.mjs
tests 8; pass 8; fail 0

$ python3 -m unittest tests.test_frontend_audio_capture tests.test_web_ui tests.test_frontend_streaming -v
Ran 10 tests ... OK

$ python3 -m unittest discover -s tests
Ran 282 tests ... OK

$ git diff --check
exit 0; no output
```

### Fix Round 1 self-review and concerns

Verified that only capture/app lifecycle and PCM byte serialization changed; `BrowserAudioOutput` remains untouched. The deferred transfer-list test wording/coverage was not broadened. No subagents or external reviewers were used. The existing physical-browser AudioWorklet smoke-test concern remains.
