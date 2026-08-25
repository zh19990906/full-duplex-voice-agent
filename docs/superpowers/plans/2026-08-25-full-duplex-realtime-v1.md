# Full Duplex Realtime V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-machine, single-user, real full-duplex voice session with backchannel handling, semantic interruption, phrase-level resume, incremental Qwen-to-CosyVoice speech, and LLM-based simultaneous interpretation.

**Architecture:** Route browser PCM through one `RealtimeSessionRuntime`. Streaming ASR and X2-Turn publish typed events; a small policy LLM and deterministic controller select actions; response and interpretation workflows publish epoch-tagged text/audio through a controllable browser player. Keep CosyVoice isolated in its worker process and avoid introducing distributed infrastructure.

**Tech Stack:** Python 3.10, asyncio, FastAPI, WebSocket, dataclasses, faster-whisper, X2-Turn, Transformers Qwen, CosyVoice worker JSONL, browser AudioWorklet, Web Audio API, unittest, Node test runner.

**Spec:** `docs/superpowers/specs/2026-08-25-full-duplex-realtime-v1-design.md`

## Global Constraints

- Deployment is one machine, one active user, one RTX PRO 5000 72GB Blackwell GPU.
- Browser service remains on port `8001` and V1 acceptance assumes a headset.
- Input audio is PCM16, 16kHz, mono, 20ms per frame.
- Persistent modes are exactly `CHAT` and `INTERPRETATION`; one-shot translation remains a CHAT request.
- User speech must duck assistant audio within 100ms; confirmed interrupt must stop old audio within 250ms.
- Turn end to first LLM token must be at most 800ms; turn end to first playable audio must be at most 1.5s.
- Stable interpretation source segment to first translated audio must be at most 2s.
- Stale audio from an old generation epoch must be zero.
- Resume may repeat or omit at most one phrase.
- V1 state recovery is process-local; restart and cross-device recovery remain out of scope.
- Use strict TDD for each task: failing test, observed failure, minimal implementation, full relevant test pass, commit.
- Do not expand `scripts/run_real_server.py` with business rules; composition belongs in `src/runtime_app` and `src/realtime`.

V1 model matrix:

| Responsibility | Model |
|---|---|
| Rolling ASR | `faster-whisper-large-v3` |
| Turn/backchannel candidates | `X2-Turn-4B-0812` |
| Semantic policy | local `1.5B–3B` Qwen Instruct selected by benchmark |
| Dialogue and translation | `Qwen2.5-14B-Instruct` |
| Streaming speech synthesis | `Fun-CosyVoice3-0.5B-2512` |

---

## File Structure

New focused modules:

```text
src/realtime/
  protocol.py              Versioned internal event envelope and audio frame header
  identifiers.py           response_id, event_id and generation epoch allocation
  session_state.py         Orthogonal mode/floor/response state and response records
  session_runtime.py       Per-session composition and lifecycle
  cancellation.py          Epoch-aware cancellation tokens and active task slots
  audio_ingress.py         PCM validation, sequence tracking and fan-out
  speech_fusion.py         ASR/turn/activity candidate fusion
  policy.py                Policy request/result contracts and timeout handling
  text_segmenter.py        Chinese/English stable TTS segmentation
  response_pipeline.py     Incremental Qwen-to-TTS orchestration
  playback.py              Server-side playback cursor and control coordination
  checkpoint.py            Pause/resume response checkpoints
  stable_prefix.py         ASR stable-prefix commitment
  interpretation.py        Persistent interpretation workflow
  supervisor.py            Worker health and one-shot restart policy

src/adapters/asr/providers/faster_whisper_streaming.py
src/adapters/turn/x2_turn_streaming.py
src/adapters/llm/providers/qwen_policy.py

frontend/src/capture-worklet.js
frontend/src/playback-worklet.js
```

Existing files to evolve rather than duplicate:

```text
src/controller/actions.py
src/controller/controller.py
src/controller/states.py
src/core/events/events.py
src/runtime/scheduler.py
src/adapters/tts/providers/cosyvoice_worker.py
scripts/cosyvoice_worker.py
src/runtime_app/bootstrap.py
src/runtime_app/container.py
scripts/run_real_server.py
frontend/src/audio.js
frontend/src/websocket.js
frontend/src/app.js
configs/models.yaml
configs/audio.yaml
README.md
```

## Dependency Order

```text
Task 1 protocol
  ├─ Task 2 state/IDs/cancellation
  ├─ Task 3 browser PCM
  └─ Task 4 browser playback
Task 2 + Task 3 → Task 5 audio ingress
Task 5 → Task 6 ASR
Task 5 → Task 7 X2/fusion
Task 6 + Task 7 → Task 8 policy/controller
Task 2 + Task 8 → Task 9 response pipeline
Task 9 → Task 10 CosyVoice cancellation
Task 4 + Task 9 + Task 10 → Task 11 pause/resume/backchannel
Task 6 + Task 9 → Task 12 interpretation
Tasks 1–12 → Task 13 real server composition
Task 13 → Task 14 reliability and acceptance benchmark
```

---

### Task 1: Freeze Realtime Protocol V1

**Files:**
- Create: `src/realtime/__init__.py`
- Create: `src/realtime/protocol.py`
- Modify: `src/core/events/events.py`
- Test: `tests/test_realtime_protocol.py`
- Docs: `docs/16-realtime-event-protocol.md`

**Interfaces:**
- Produces: `RealtimeEnvelope`, `AudioFrameHeader`, `encode_audio_frame()`, `decode_audio_frame()`.
- Consumes: no new project interface.

- [ ] **Step 1: Write failing envelope and binary frame tests**

```python
def test_envelope_contains_ordering_and_generation_identity():
    envelope = RealtimeEnvelope(
        event="BACKCHANNEL_CONFIRMED",
        event_id="evt-1",
        session_id="session-1",
        sequence=4,
        capture_timestamp=10.0,
        server_timestamp=10.1,
        response_id="response-2",
        generation_epoch=3,
        payload={"confidence": 0.9},
    )
    assert envelope.to_dict()["protocol_version"] == 1
    assert envelope.to_dict()["generation_epoch"] == 3


def test_audio_frame_round_trip_preserves_pcm_and_header():
    header = AudioFrameHeader(sequence=8, capture_timestamp=4.5, sample_rate=16000, channels=1)
    encoded = encode_audio_frame(header, b"\x01\x00\x02\x00")
    decoded_header, pcm = decode_audio_frame(encoded)
    assert decoded_header == header
    assert pcm == b"\x01\x00\x02\x00"
```

- [ ] **Step 2: Run the test and observe the missing module failure**

Run: `python -m unittest tests.test_realtime_protocol -v`

Expected: FAIL because `src.realtime.protocol` does not exist.

- [ ] **Step 3: Implement the exact protocol contracts**

```python
@dataclass(frozen=True)
class RealtimeEnvelope:
    event: str
    event_id: str
    session_id: str
    sequence: int
    capture_timestamp: float
    server_timestamp: float
    response_id: str | None
    generation_epoch: int
    payload: dict[str, Any]
    protocol_version: int = 1


@dataclass(frozen=True)
class AudioFrameHeader:
    sequence: int
    capture_timestamp: float
    sample_rate: int = 16000
    channels: int = 1
```

Use one documented `struct.Struct` layout for the binary header. Reject unsupported protocol versions, non-16kHz input, non-mono input, odd PCM byte lengths and truncated frames with `ValueError`.

- [ ] **Step 4: Update protocol documentation and run tests**

Run: `python -m unittest tests.test_realtime_protocol tests.test_core_contracts tests.test_api_service -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/realtime src/core/events/events.py tests/test_realtime_protocol.py docs/16-realtime-event-protocol.md
git commit -m "feat: freeze realtime protocol v1"
```

---

### Task 2: Add Orthogonal Session State, IDs, and Preemptible Task Slots

**Files:**
- Create: `src/realtime/identifiers.py`
- Create: `src/realtime/session_state.py`
- Create: `src/realtime/cancellation.py`
- Modify: `src/controller/states.py`
- Modify: `src/runtime/scheduler.py`
- Test: `tests/test_realtime_session_state.py`
- Test: `tests/test_runtime_preemption.py`

**Interfaces:**
- Produces: `ConversationMode`, `FloorState`, `ResponseState`, `ResponseRecord`, `GenerationClock`, `ActiveTaskSlot`, deadline-aware scheduler priorities.
- Consumes: Task 1 `RealtimeEnvelope` identifiers.

- [ ] **Step 1: Write failing state and epoch tests**

```python
def test_generation_clock_invalidates_old_epoch():
    clock = GenerationClock()
    first = clock.current
    second = clock.advance()
    assert second == first + 1
    assert not clock.is_current(first)


async def test_active_task_slot_preempts_running_task():
    slot = ActiveTaskSlot()
    stopped = asyncio.Event()

    async def old_work():
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    await slot.replace(old_work())
    await slot.replace(asyncio.sleep(0))
    await asyncio.wait_for(stopped.wait(), 0.1)
```

- [ ] **Step 2: Run tests and observe missing contract failures**

Run: `python -m unittest tests.test_realtime_session_state tests.test_runtime_preemption -v`

Expected: FAIL because the new classes do not exist.

- [ ] **Step 3: Implement state and cancellation primitives**

```python
class ConversationMode(str, Enum):
    CHAT = "CHAT"
    INTERPRETATION = "INTERPRETATION"


class FloorState(str, Enum):
    NONE = "NONE"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    OVERLAP = "OVERLAP"


class ResponseState(str, Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    PLAYING = "PLAYING"
    DUCKED = "DUCKED"
    PAUSED = "PAUSED"
    CANCELLING = "CANCELLING"
```

`ActiveTaskSlot.replace()` must cancel and await the prior task before installing the next task. `GenerationClock` starts at zero and advances monotonically.

Assign scheduler priority in this exact order: playback stop/duck, speech-start and X2, semantic policy, first TTS segment, main LLM, later TTS/background state. Every scheduled item carries a monotonic deadline and cancellation token; workflow-owned `ActiveTaskSlot` instances provide preemption for already-running asyncio tasks.

- [ ] **Step 4: Run state, scheduler and existing runtime tests**

Run: `python -m unittest tests.test_realtime_session_state tests.test_runtime_preemption tests.test_runtime tests.test_runtime_scheduler -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/realtime src/controller/states.py src/runtime/scheduler.py tests/test_realtime_session_state.py tests/test_runtime_preemption.py
git commit -m "feat: add preemptible realtime session state"
```

---

### Task 3: Replace Browser MediaRecorder Input with PCM AudioWorklet

**Files:**
- Create: `frontend/src/capture-worklet.js`
- Modify: `frontend/src/audio.js`
- Modify: `frontend/src/websocket.js`
- Modify: `frontend/src/app.js`
- Test: `tests/frontend_audio_capture_test.mjs`
- Test: `tests/test_frontend_audio_capture.py`

**Interfaces:**
- Produces: `PcmMicrophoneInput.start()`, `PcmMicrophoneInput.stop()`, 20ms PCM16 frames with sequence and capture timestamp.
- Consumes: Task 1 binary audio frame layout.

- [ ] **Step 1: Write a failing Node behavior test for PCM conversion and sequencing**

```javascript
test("float samples become ordered PCM16 frames", () => {
  const encoder = new PcmFrameEncoder({ sampleRate: 16000, frameSamples: 320 });
  const first = encoder.encode(new Float32Array([0, 1, -1]));
  const second = encoder.encode(new Float32Array([0.5]));
  assert.equal(first.sequence, 0);
  assert.equal(second.sequence, 1);
  assert.deepEqual([...first.pcm], [0, 32767, -32768]);
});
```

- [ ] **Step 2: Run the test and observe missing encoder failure**

Run: `node --test tests/frontend_audio_capture_test.mjs`

Expected: FAIL because `PcmFrameEncoder` and capture worklet do not exist.

- [ ] **Step 3: Implement capture worklet and microphone lifecycle**

Use `getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`. Resample to 16kHz in the worklet, buffer exactly 320 samples, clamp to signed PCM16 and post transferable `ArrayBuffer` frames to the main thread.

- [ ] **Step 4: Run browser module and existing UI tests**

Run: `python -m unittest tests.test_frontend_audio_capture tests.test_web_ui tests.test_frontend_streaming -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/capture-worklet.js frontend/src/audio.js frontend/src/websocket.js frontend/src/app.js tests/frontend_audio_capture_test.mjs tests/test_frontend_audio_capture.py
git commit -m "feat: stream browser microphone pcm frames"
```

---

### Task 4: Build a Duckable, Acknowledged Browser Playback Queue

**Files:**
- Create: `frontend/src/playback-worklet.js`
- Modify: `frontend/src/audio.js`
- Modify: `frontend/src/websocket.js`
- Modify: `frontend/src/app.js`
- Test: `tests/frontend_playback_test.mjs`
- Test: `tests/test_frontend_playback.py`

**Interfaces:**
- Produces: `BrowserPlaybackCoordinator.enqueue()`, `duck()`, `restore()`, `pauseResponse()`, `stopResponse()` and playback ACK messages.
- Consumes: Task 1 response/epoch/segment metadata.

- [ ] **Step 1: Write failing queue identity and stale epoch tests**

```javascript
test("player rejects old epochs and stops one response", () => {
  const queue = new PlaybackQueue();
  queue.setEpoch(4);
  assert.equal(queue.enqueue({ responseId: "r1", epoch: 3, segmentId: 1 }), false);
  assert.equal(queue.enqueue({ responseId: "r1", epoch: 4, segmentId: 2 }), true);
  queue.stopResponse("r1");
  assert.equal(queue.pending.length, 0);
});
```

- [ ] **Step 2: Run and observe missing playback coordinator failure**

Run: `node --test tests/frontend_playback_test.mjs`

Expected: FAIL because `PlaybackQueue` does not exist.

- [ ] **Step 3: Implement queue, GainNode controls and ACK**

Duck gain must ramp to the configured level within 100ms. ACK payload must include `response_id`, `generation_epoch`, `segment_id`, `sample_offset` and browser audio clock time. Stop source nodes by identity; do not close the entire AudioContext for a normal interrupt.

- [ ] **Step 4: Run playback and UI regression tests**

Run: `python -m unittest tests.test_frontend_playback tests.test_web_ui tests.test_frontend_streaming -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playback-worklet.js frontend/src/audio.js frontend/src/websocket.js frontend/src/app.js tests/frontend_playback_test.mjs tests/test_frontend_playback.py
git commit -m "feat: add controllable browser playback queue"
```

---

### Task 5: Implement PCM Audio Ingress and Fan-Out

**Files:**
- Create: `src/realtime/audio_ingress.py`
- Create: `src/realtime/session_runtime.py`
- Test: `tests/test_realtime_audio_ingress.py`

**Interfaces:**
- Produces: `AudioIngress.push(frame: bytes) -> None`, `AudioActivityCandidate`.
- Consumes: Task 1 `decode_audio_frame()` and Task 2 session epoch.

- [ ] **Step 1: Write failing ordered fan-out tests**

```python
async def test_ingress_fans_ordered_pcm_to_all_consumers():
    received = {"asr": [], "turn": []}
    ingress = AudioIngress(
        asr_consumer=lambda frame: received["asr"].append(frame.sequence),
        turn_consumer=lambda frame: received["turn"].append(frame.sequence),
    )
    later = encode_audio_frame(AudioFrameHeader(sequence=1, capture_timestamp=0.02), b"\x00\x00" * 320)
    earlier = encode_audio_frame(AudioFrameHeader(sequence=0, capture_timestamp=0.00), b"\x00\x00" * 320)
    await ingress.push(later)
    await ingress.push(earlier)
    await ingress.flush()
    assert received == {"asr": [0, 1], "turn": [0, 1]}
```

- [ ] **Step 2: Run and observe missing ingress failure**

Run: `python -m unittest tests.test_realtime_audio_ingress -v`

Expected: FAIL because `AudioIngress` does not exist.

- [ ] **Step 3: Implement bounded jitter buffering and activity candidates**

Reject duplicate frames, record missing sequences, bound the jitter buffer, and call both consumers without allowing one slow consumer to block the other. Use bounded asyncio queues and expose overflow metrics.

- [ ] **Step 4: Run ingress and audio pipeline tests**

Run: `python -m unittest tests.test_realtime_audio_ingress tests.test_audio_pipeline tests.test_audio_stream -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/realtime/audio_ingress.py src/realtime/session_runtime.py tests/test_realtime_audio_ingress.py
git commit -m "feat: add realtime pcm audio ingress"
```

---

### Task 6: Add faster-whisper Rolling ASR and Stable Partials

**Files:**
- Create: `src/adapters/asr/providers/faster_whisper_streaming.py`
- Create: `src/realtime/stable_prefix.py`
- Modify: `src/asr/stream.py`
- Test: `tests/test_faster_whisper_streaming.py`
- Test: `tests/test_stable_prefix.py`
- Script: `scripts/benchmark_streaming_asr.py`

**Interfaces:**
- Produces: `FasterWhisperStreamingProvider.push_pcm()`, `StablePrefixCommitter.update()`, revision-aware `TranscriptChunk`.
- Consumes: Task 5 ordered 16kHz PCM frames.

- [ ] **Step 1: Write failing stable prefix tests**

```python
def test_committer_emits_only_new_common_prefix():
    committer = StablePrefixCommitter()
    assert committer.update("我想去北") == ""
    assert committer.update("我想去北京") == "我想去北"
    assert committer.update("我想去北京旅游") == "京"
```

Add a provider test with an injected fake runtime that verifies decode cadence, rolling context and final decode on turn end.

- [ ] **Step 2: Run and observe missing provider/committer failures**

Run: `python -m unittest tests.test_stable_prefix tests.test_faster_whisper_streaming -v`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement rolling decode and revision-aware chunks**

Decode every configured 200–400ms, retain a bounded audio context, publish only new committed text, keep an unstable tail and force final decode when `finalize_turn()` is called.

- [ ] **Step 4: Run model-independent tests and add benchmark CLI validation**

Run: `python -m unittest tests.test_stable_prefix tests.test_faster_whisper_streaming tests.test_asr_pipeline tests.test_asr_provider -v`

Expected: PASS. `python scripts/benchmark_streaming_asr.py --help` must exit zero without loading a model.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/asr/providers/faster_whisper_streaming.py src/realtime/stable_prefix.py src/asr/stream.py tests/test_faster_whisper_streaming.py tests/test_stable_prefix.py scripts/benchmark_streaming_asr.py
git commit -m "feat: add rolling faster whisper asr"
```

---

### Task 7: Add X2-Turn Rolling Worker and Speech Event Fusion

**Files:**
- Create: `src/adapters/turn/x2_turn_streaming.py`
- Create: `src/realtime/speech_fusion.py`
- Modify: `src/adapters/turn/x2_turn_adapter.py`
- Test: `tests/test_x2_turn_streaming.py`
- Test: `tests/test_speech_event_fusion.py`
- Script: `scripts/benchmark_x2_turn_streaming.py`

**Interfaces:**
- Produces: `TurnCandidate`, `X2TurnRollingProvider.push_pcm()`, `SpeechEventFusion.accept_*()`.
- Consumes: Task 5 PCM frames and Task 6 transcript chunks.

- [ ] **Step 1: Write failing hysteresis and candidate tests**

```python
def test_turn_end_requires_consecutive_confirmation_frames():
    fusion = SpeechEventFusion(turn_end_frames=2)
    assert fusion.accept_turn(TurnCandidate("turn_end", 0.8)) == ()
    events = fusion.accept_turn(TurnCandidate("turn_end", 0.9))
    assert [event.event for event in events] == ["USER_TURN_END_CANDIDATE"]


def test_backchannel_remains_candidate_until_text_confirmation():
    fusion = SpeechEventFusion()
    events = fusion.accept_turn(TurnCandidate("backchannel", 0.9))
    assert events[0].event == "USER_BACKCHANNEL_CANDIDATE"
```

- [ ] **Step 2: Run and observe missing rolling/fusion modules**

Run: `python -m unittest tests.test_x2_turn_streaming tests.test_speech_event_fusion -v`

Expected: FAIL.

- [ ] **Step 3: Implement sliding context, cadence and hysteresis**

Maintain 1–3 seconds of PCM context, run injected X2 inference every 100–200ms, map frame labels to candidates, and expose inference duration/RTF. Do not emit final semantic actions from this adapter.

- [ ] **Step 4: Run turn adapter and fusion tests**

Run: `python -m unittest tests.test_x2_turn_streaming tests.test_speech_event_fusion tests.test_x2_turn_adapter -v`

Expected: PASS. Benchmark CLI help must exit zero without model loading.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/turn/x2_turn_streaming.py src/adapters/turn/x2_turn_adapter.py src/realtime/speech_fusion.py tests/test_x2_turn_streaming.py tests/test_speech_event_fusion.py scripts/benchmark_x2_turn_streaming.py
git commit -m "feat: add rolling x2 turn fusion"
```

---

### Task 8: Add Semantic Policy LLM and Upgrade Controller Actions

**Files:**
- Create: `src/adapters/llm/providers/qwen_policy.py`
- Create: `src/realtime/policy.py`
- Modify: `src/controller/actions.py`
- Modify: `src/controller/controller.py`
- Modify: `src/controller/states.py`
- Modify: `configs/models.yaml`
- Test: `tests/test_semantic_policy.py`
- Test: `tests/test_conversation_controller_v1.py`
- Script: `scripts/benchmark_policy_llm.py`

**Interfaces:**
- Produces: `PolicyRequest`, `PolicyDecision`, `PolicyAction`, `SemanticPolicyEngine.decide()`.
- Consumes: Task 2 state and Task 7 speech candidates.

- [ ] **Step 1: Write failing policy parsing and contextual affirmation tests**

```python
def test_policy_rejects_unknown_action(self):
    with self.assertRaises(ValueError):
        PolicyDecision.from_mapping({"action": "GUESS", "confidence": 1.0})


def test_affirmation_after_question_is_answer_not_backchannel():
    controller = ConversationController()
    state = SessionState(
        mode=ConversationMode.CHAT,
        floor=FloorState.OVERLAP,
        response=ResponseState.DUCKED,
        assistant_act="ASKING",
    )
    policy = PolicyDecision(
        action=PolicyAction.ANSWER,
        confidence=0.98,
        rationale="The user answered the assistant's question.",
    )
    actions = controller.apply_policy(state, policy)
    assert [item.action_type for item in actions] == [ActionType.STOP_RESPONSE, ActionType.PROCESS_USER_REQUEST]
```

Place these methods on a `unittest.TestCase`; the exact assertions above are the required contract.

- [ ] **Step 2: Run and observe missing schema/action failures**

Run: `python -m unittest tests.test_semantic_policy tests.test_conversation_controller_v1 -v`

Expected: FAIL.

- [ ] **Step 3: Implement strict policy schema and two-stage action handling**

Allowed actions are exactly `BACKCHANNEL`, `ANSWER`, `PAUSE`, `RESUME`, `REVISE`, `NEW_REQUEST`, `MODE_SWITCH`, `UNCERTAIN`. Enforce a timeout with `asyncio.wait_for`; map timeout, malformed JSON and low confidence to `UNCERTAIN`. Add controller actions for duck, restore, pause, resume, revise, switch mode and clarification.

Configure a separate local Qwen Instruct policy model in the 1.5B–3B range. `PolicyDecision` has `action`, `confidence`, `rationale`, optional `intent`, optional `source_language` and optional `target_language`; reject unknown keys and actions. The benchmark script compares GPU and CPU/quantized placement and reports p50/p95 latency; V1 requires policy p95 at or below 300ms and chooses placement from that evidence.

- [ ] **Step 4: Run policy, controller and existing scenario tests**

Run: `python -m unittest tests.test_semantic_policy tests.test_conversation_controller_v1 tests.test_conversation_controller tests.test_full_duplex_integration -v`

Expected: PASS. `python scripts/benchmark_policy_llm.py --help` must exit zero without loading a model.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/llm/providers/qwen_policy.py src/realtime/policy.py src/controller configs/models.yaml tests/test_semantic_policy.py tests/test_conversation_controller_v1.py scripts/benchmark_policy_llm.py
git commit -m "feat: add semantic realtime policy controller"
```

---

### Task 9: Stream Qwen Text into Stable TTS Segments with Epoch Filtering

**Files:**
- Create: `src/realtime/text_segmenter.py`
- Create: `src/realtime/response_pipeline.py`
- Modify: `src/runtime_app/real_session.py`
- Test: `tests/test_text_segmenter.py`
- Test: `tests/test_realtime_response_pipeline.py`

**Interfaces:**
- Produces: `TextSegment`, `LanguageAwareTextSegmenter.push()`, `RealtimeResponsePipeline.run()`.
- Consumes: Task 2 epoch clock and existing Qwen token provider.

- [ ] **Step 1: Write failing Chinese segmentation and stale token tests**

```python
def test_chinese_first_segment_commits_at_clause_boundary():
    segmenter = LanguageAwareTextSegmenter(first_min_chars=12)
    assert segmenter.push("北京是一座历史悠久的城市，") == (
        TextSegment(segment_id=0, text="北京是一座历史悠久的城市，", is_final=False),
    )


async def test_old_epoch_tokens_never_reach_tts_queue():
    clock = GenerationClock()
    first_token_seen = asyncio.Event()
    llm = ControlledTokenProvider(
        tokens=("旧", "回答", "。"),
        first_token_seen=first_token_seen,
        release_after_first=asyncio.Event(),
    )
    tts_queue = RecordingSegmentQueue()
    pipeline = RealtimeResponsePipeline(
        llm=llm,
        segment_queue=tts_queue,
        generation_clock=clock,
    )
    task = asyncio.create_task(pipeline.run("问题"))
    await first_token_seen.wait()
    clock.advance()
    llm.release_after_first.set()
    await task
    assert tts_queue.items == []
```

Define `ControlledTokenProvider` and `RecordingSegmentQueue` in the test file as small fakes implementing the production provider/queue protocols; they must not import a real model.

- [ ] **Step 2: Run and observe missing segmenter/pipeline failures**

Run: `python -m unittest tests.test_text_segmenter tests.test_realtime_response_pipeline -v`

Expected: FAIL.

- [ ] **Step 3: Implement language-aware segmentation and bounded queues**

Commit Chinese on punctuation or configured first/next character thresholds; commit English on clause punctuation or word thresholds. Tag every segment with response/epoch/segment identity. Stop publishing immediately when epoch changes. Use a bounded asyncio queue and await capacity to apply backpressure.

- [ ] **Step 4: Run response and existing Qwen session tests**

Run: `python -m unittest tests.test_text_segmenter tests.test_realtime_response_pipeline tests.test_real_session tests.test_qwen_transformers_provider -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/realtime/text_segmenter.py src/realtime/response_pipeline.py src/runtime_app/real_session.py tests/test_text_segmenter.py tests/test_realtime_response_pipeline.py
git commit -m "feat: stream qwen responses into tts segments"
```

---

### Task 10: Add CosyVoice Request Cancellation and Identity

**Files:**
- Modify: `src/adapters/tts/providers/cosyvoice_worker.py`
- Modify: `scripts/cosyvoice_worker.py`
- Test: `tests/test_cosyvoice_worker.py`
- Test: `tests/test_cosyvoice_worker_script.py`

**Interfaces:**
- Produces: `CosyVoiceWorkerClient.cancel(request_id)`, request-tagged `AudioChunk`.
- Consumes: Task 9 text segments.

- [ ] **Step 1: Write failing cancel protocol tests**

```python
def test_worker_cancel_message_contains_request_identity():
    assert json.loads(encode_worker_cancel("request-7")) == {
        "op": "cancel",
        "request_id": "request-7",
    }


async def test_cancelled_request_discards_late_audio():
    transport = ScriptedWorkerTransport(
        messages=[
            {"event": "audio", "request_id": "request-7", "pcm": "AAAA"},
            {"event": "done", "request_id": "request-7"},
        ]
    )
    client = CosyVoiceWorkerClient(transport=transport)
    stream = await client.stream_audio("文本", request_id="request-7")
    await client.cancel("request-7")
    assert [chunk async for chunk in stream] == []
```

Define `ScriptedWorkerTransport` in the test against the worker transport protocol; it returns the listed JSON messages after cancellation so the stale-chunk filter is exercised without starting CosyVoice.

- [ ] **Step 2: Run and observe missing cancel support**

Run: `python -m unittest tests.test_cosyvoice_worker tests.test_cosyvoice_worker_script -v`

Expected: FAIL for missing cancel encoding/client API.

- [ ] **Step 3: Implement cancel, drain and request identity**

Worker must accept `synthesize`, `cancel` and `shutdown`. The client must serialize writes, maintain cancelled request IDs, ignore late chunks, and clear request state on terminal messages. Cooperative worker cancellation occurs between yielded chunks; epoch filtering remains the final safety boundary.

- [ ] **Step 4: Run all CosyVoice worker tests**

Run: `python -m unittest tests.test_cosyvoice_worker tests.test_cosyvoice_worker_script tests.test_cosyvoice_worker_smoke_script tests.test_tts_provider -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/tts/providers/cosyvoice_worker.py scripts/cosyvoice_worker.py tests/test_cosyvoice_worker.py tests/test_cosyvoice_worker_script.py
git commit -m "feat: cancel cosyvoice synthesis requests"
```

---

### Task 11: Implement Playback Checkpoints, Backchannel, Pause, and Resume

**Files:**
- Create: `src/realtime/playback.py`
- Create: `src/realtime/checkpoint.py`
- Modify: `src/realtime/session_runtime.py`
- Modify: `src/controller/controller.py`
- Test: `tests/test_playback_coordinator.py`
- Test: `tests/test_response_checkpoint.py`
- Test: `tests/test_real_backchannel_resume.py`

**Interfaces:**
- Produces: `PlaybackAck`, `PlaybackCoordinator`, `ResponseCheckpointStore.pause()`, `resume()`.
- Consumes: Tasks 4, 8, 9 and 10.

- [ ] **Step 1: Write failing phrase-resume and backchannel tests**

```python
def test_partial_segment_resumes_from_segment_start():
    store = ResponseCheckpointStore()
    store.record_segment("r1", 4, "第二个是故宫。", audio=b"pcm")
    store.ack("r1", segment_id=4, sample_offset=1800)
    plan = store.resume("r1")
    assert plan.segment_id == 4
    assert plan.sample_offset == 0


async def test_backchannel_restores_without_cancelling_generation():
    clock = GenerationClock()
    commands = []
    playback = PlaybackCoordinator(send_command=commands.append)
    controller = ConversationController()
    state = SessionState(
        mode=ConversationMode.CHAT,
        floor=FloorState.OVERLAP,
        response=ResponseState.DUCKED,
        assistant_act="EXPLAINING",
    )
    policy = PolicyDecision(
        action=PolicyAction.BACKCHANNEL,
        confidence=0.97,
        rationale="Short acknowledgement while assistant is explaining.",
    )
    for action in controller.apply_policy(state, policy):
        await playback.apply(action)
    assert clock.current == 0
    assert commands == ["RESTORE"]
```

- [ ] **Step 2: Run and observe missing checkpoint/coordinator failures**

Run: `python -m unittest tests.test_playback_coordinator tests.test_response_checkpoint tests.test_real_backchannel_resume -v`

Expected: FAIL.

- [ ] **Step 3: Implement four cursors and action semantics**

Track generated, committed, synthesized and played cursors. `PAUSE` preserves text and reusable audio; `RESUME` starts the partially played segment from offset zero; `REVISE` advances epoch and discards unplayed content; `NEW_REQUEST` archives the old response. A confirmed backchannel restores gain without adding a formal user message.

- [ ] **Step 4: Run interruption, generation and integration tests**

Run: `python -m unittest tests.test_playback_coordinator tests.test_response_checkpoint tests.test_real_backchannel_resume tests.test_generation_manager tests.test_full_duplex_integration -v`

Expected: PASS with zero stale output assertions.

- [ ] **Step 5: Commit**

```bash
git add src/realtime/playback.py src/realtime/checkpoint.py src/realtime/session_runtime.py src/controller/controller.py tests/test_playback_coordinator.py tests/test_response_checkpoint.py tests/test_real_backchannel_resume.py
git commit -m "feat: resume interrupted responses by phrase"
```

---

### Task 12: Implement LLM-Based Interpretation Mode

**Files:**
- Create: `src/realtime/interpretation.py`
- Modify: `src/realtime/stable_prefix.py`
- Modify: `src/realtime/session_runtime.py`
- Modify: `src/realtime/policy.py`
- Test: `tests/test_interpretation_runtime.py`
- Test: `tests/test_interpretation_mode_policy.py`

**Interfaces:**
- Produces: `InterpretationSession`, `SourceSegment`, `TranslationSegment`, `InterpretationPipeline.push_partial()`.
- Consumes: Tasks 6, 8, 9 and 10.

- [ ] **Step 1: Write failing one-shot/persistent mode and deduplication tests**

```python
def test_one_shot_translation_does_not_switch_mode():
    state = SessionState(
        mode=ConversationMode.CHAT,
        floor=FloorState.USER,
        response=ResponseState.IDLE,
    )
    controller = ConversationController()
    policy = PolicyDecision(
        action=PolicyAction.ANSWER,
        confidence=0.99,
        intent="translate_once",
        rationale="A single translation request is ordinary chat.",
    )
    controller.apply_policy(state, policy)
    assert state.mode is ConversationMode.CHAT


async def test_interpretation_translates_each_stable_source_once():
    translator = RecordingTranslator(result="I am going to Beijing tomorrow.")
    sink = RecordingTranslationSink()
    pipeline = InterpretationPipeline(translator=translator, sink=sink, target_language="English")
    await pipeline.push_partial("我明天去北京")
    await pipeline.push_partial("我明天去北京")
    await pipeline.push_partial("我明天去北京")
    assert translator.source_texts == ["我明天去北京"]
    assert [item.source_text for item in sink.items] == ["我明天去北京"]
```

Define `RecordingTranslator` and `RecordingTranslationSink` in the test file against the production translation protocols so this test remains model-independent.

- [ ] **Step 2: Run and observe missing interpretation workflow failure**

Run: `python -m unittest tests.test_interpretation_runtime tests.test_interpretation_mode_policy -v`

Expected: FAIL.

- [ ] **Step 3: Implement persistent mode and strict translation prompts**

Mode switches only on a `MODE_SWITCH` decision. One-shot translation stays in CHAT. Maintain source segment IDs, translation segment IDs and target language. Never revise audio that has begun playback; apply target-language changes from the next stable source segment.

- [ ] **Step 4: Run interpretation and existing translation tests**

Run: `python -m unittest tests.test_interpretation_runtime tests.test_interpretation_mode_policy tests.test_translation_pipeline -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/realtime/interpretation.py src/realtime/stable_prefix.py src/realtime/session_runtime.py src/realtime/policy.py tests/test_interpretation_runtime.py tests/test_interpretation_mode_policy.py
git commit -m "feat: add llm interpretation mode"
```

---

### Task 13: Replace the Serial Real Server Path with `RealtimeSessionRuntime`

**Files:**
- Modify: `src/runtime_app/bootstrap.py`
- Modify: `src/runtime_app/container.py`
- Modify: `src/model_runtime/factory.py`
- Modify: `scripts/run_real_server.py`
- Modify: `frontend/src/app.js`
- Modify: `configs/models.yaml`
- Modify: `configs/audio.yaml`
- Test: `tests/test_realtime_bootstrap.py`
- Test: `tests/test_real_server_script.py`
- Test: `tests/test_realtime_websocket_flow.py`

**Interfaces:**
- Produces: `RealtimeServerSettings`, one production composition root and browser WebSocket flow.
- Consumes: Tasks 1–12.

- [ ] **Step 1: Write failing composition and WebSocket flow tests**

```python
def test_binary_audio_enters_runtime_and_audio_events_return():
    runtime = RecordingRealtimeRuntime()
    settings = RealtimeServerSettings(static_dir=FRONTEND_DIR)
    app = build_app(settings, runtime_factory=lambda session_id: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            frame = encode_audio_frame(
                AudioFrameHeader(sequence=0, capture_timestamp=0.0),
                b"\x00\x00" * 320,
            )
            websocket.send_bytes(frame)
            assert websocket.receive_json()["event"] == "AUDIO_FRAME_ACCEPTED"
    assert runtime.audio_sequences == [0]


def test_real_server_has_no_direct_qwen_to_tts_session():
    settings = RealtimeServerSettings(static_dir=FRONTEND_DIR)
    app = build_app(settings, runtime_factory=recording_runtime_factory)
    assert app.state.runtime_factory is recording_runtime_factory
```

`RecordingRealtimeRuntime` in the test implements `start()`, `accept_audio_frame()`, `accept_command()`, `events()` and `close()`; `accept_audio_frame()` records the decoded sequence and publishes `AUDIO_FRAME_ACCEPTED`. This is the same runtime protocol used by production composition.

- [ ] **Step 2: Run and observe serial path failures**

Run: `python -m unittest tests.test_realtime_bootstrap tests.test_real_server_script tests.test_realtime_websocket_flow -v`

Expected: FAIL because the server still creates `RealModelSession` directly.

- [ ] **Step 3: Wire one composition root and externalize real model paths**

Build workers from configuration, create one `RealtimeSessionRuntime` per active session, route binary frames to `AudioIngress`, route text/control messages to runtime commands and publish runtime wire events. Remove the global model lock and direct `session.model.run()` business path. Keep FastAPI lifespan cleanup.

- [ ] **Step 4: Run server, bootstrap, frontend and full unit suite**

Run: `python -m unittest tests.test_realtime_bootstrap tests.test_real_server_script tests.test_realtime_websocket_flow tests.test_web_ui -v`

Then run: `python -m unittest discover -s tests -p 'test_*.py' -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/runtime_app src/model_runtime/factory.py scripts/run_real_server.py frontend/src/app.js configs/models.yaml configs/audio.yaml tests/test_realtime_bootstrap.py tests/test_real_server_script.py tests/test_realtime_websocket_flow.py
git commit -m "feat: run real models through realtime runtime"
```

---

### Task 14: Add Worker Supervision, Reconnect Snapshot, and Real Acceptance Gates

**Files:**
- Create: `src/realtime/supervisor.py`
- Modify: `src/realtime/session_runtime.py`
- Modify: `benchmarks/real_hardware_runner.py`
- Modify: `benchmarks/metrics.py`
- Modify: `src/model_runtime/manager.py`
- Modify: `scripts/check_environment.sh`
- Create: `scripts/run_realtime_acceptance.py`
- Modify: `README.md`
- Modify: `docs/hardware_benchmark.md`
- Test: `tests/test_realtime_supervisor.py`
- Test: `tests/test_realtime_snapshot.py`
- Test: `tests/test_realtime_acceptance.py`
- Test: `tests/test_realtime_degradation.py`
- Test: `tests/test_model_memory_budget.py`

**Interfaces:**
- Produces: `WorkerSupervisor`, reconnect session snapshot, real acceptance report.
- Consumes: complete runtime from Task 13.

- [ ] **Step 1: Write failing restart, snapshot and metric threshold tests**

```python
async def test_worker_restarts_once_then_surfaces_failure():
    worker = AlwaysFailingWorker()
    supervisor = WorkerSupervisor(max_restarts=1)
    await supervisor.run(worker)
    assert worker.start_count == 2
    assert supervisor.status == "FAILED"


def test_acceptance_rejects_stale_audio_and_slow_interrupt():
    result = evaluate_acceptance({
        "interrupt_latency_ms": 251.0,
        "stale_output_count": 1.0,
    })
    assert not result.passed
    assert set(result.failures) == {"interrupt_latency_ms", "stale_output_count"}


def test_model_manager_rejects_load_above_vram_budget(self):
    manager = ModelManager(total_vram_bytes=72 * 1024**3, reserve_bytes=10 * 1024**3)
    manager.record_loaded("qwen", used_bytes=58 * 1024**3)
    with self.assertRaises(ModelMemoryBudgetError):
        manager.reserve("cosyvoice", requested_bytes=8 * 1024**3)
```

Define `AlwaysFailingWorker.start()` in the test to increment `start_count` and raise `RuntimeError`; no subprocess or model is required.

- [ ] **Step 2: Run and observe missing supervision/acceptance failures**

Run: `python -m unittest tests.test_realtime_supervisor tests.test_realtime_snapshot tests.test_realtime_acceptance tests.test_realtime_degradation tests.test_model_memory_budget -v`

Expected: FAIL.

- [ ] **Step 3: Implement one-shot restart, process-local snapshot and hard gates**

Restart a failed worker at most once, restore unplayed text for TTS restart, and surface terminal failures to the browser. Snapshot must include mode, floor, response state, epoch, current response, paused responses and playback cursor. Acceptance evaluator must enforce all thresholds from the spec and label reports `hardware-e2e`.

Record CUDA memory before and after each model load, reserve explicit space for KV cache/CUDA context/temporary tensors, and reject a load that exceeds the configured safe budget with per-model diagnostics. Implement the spec's degradation matrix: X2 timeout falls back to activity plus ASR stability; ASR failure asks the user to repeat; policy timeout safely pauses as `UNCERTAIN`; LLM/TTS failures preserve state or text; playback failure stops further audio enqueue. Make stop, cancel and close idempotent and cover each path in `tests/test_realtime_degradation.py`.

- [ ] **Step 4: Run full verification**

Run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
./scripts/run_demo.sh
python scripts/run_realtime_acceptance.py --help
```

Expected: unit suite and demo PASS; CLI help exits zero. On the GPU server, run the documented recorded-audio and headset sessions and require every V1 gate to pass before tagging the milestone.

- [ ] **Step 5: Commit**

```bash
git add src/realtime/supervisor.py src/realtime/session_runtime.py src/model_runtime/manager.py benchmarks scripts/check_environment.sh scripts/run_realtime_acceptance.py README.md docs/hardware_benchmark.md tests/test_realtime_supervisor.py tests/test_realtime_snapshot.py tests/test_realtime_acceptance.py tests/test_realtime_degradation.py tests/test_model_memory_budget.py
git commit -m "feat: gate realtime v1 on hardware acceptance"
```

---

## Final Release Verification

- [ ] Run all unit and simulated integration tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

- [ ] Run deterministic demo scenarios:

```bash
./scripts/run_demo.sh
```

- [ ] Run model integration with recorded audio on the GPU server:

```bash
python scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --report /tmp/realtime-model-integration.json
```

- [ ] Run headset hardware E2E and confirm:

```text
duck_latency_ms <= 100
interrupt_latency_ms <= 250
backchannel_restore_latency_ms <= 300
first_token_latency_ms <= 800
first_audio_latency_ms <= 1500
first_translated_audio_latency_ms <= 2000
stale_output_count == 0
resume_phrase_error_count <= 1
```

- [ ] Run a 60-minute soak session and verify no unbounded queue growth, worker restart loop, CUDA OOM or stale audio.

- [ ] Record the final capability matrix with explicit labels: `unit`, `simulated`, `model-integration`, `hardware-e2e`.
