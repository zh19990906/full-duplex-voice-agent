# Task 6: Rolling faster-whisper ASR and Stable Partials

## RED evidence

Before implementation, the required new tests were run with:

```text
python3 -m unittest tests.test_stable_prefix tests.test_faster_whisper_streaming -v
```

They failed as intended because `src.realtime.stable_prefix` and
`src.adapters.asr.providers.faster_whisper_streaming` did not exist.

## Design

- `StablePrefixCommitter` commits only the newly added part of the longest
  common prefix from consecutive hypotheses. It keeps append-only committed
  text plus an explicitly replaceable unstable tail, and finalization emits
  the remaining suffix once before reset.
- `TranscriptChunk` retains its original four positional fields and appends
  `revision_id` and `unstable_text` with backward-compatible defaults.
- `FasterWhisperStreamingProvider` consumes V1 20 ms PCM16 frames (raw bytes
  or `RealtimeAudioFrame`), validates 16 kHz mono metadata, retains a bounded
  context, and decodes every configured 200--400 ms.
- Every turn is serialized with an async lock. Synchronous runtimes run through
  `asyncio.to_thread`; async runtimes remain awaitable. Cancellation advances a
  generation marker so an in-flight decoder result cannot publish late text.
- Runtime injection is the default. The optional faster-whisper import/model
  construction occurs only with `load_model=True`; the benchmark CLI imports
  the provider only after argument parsing.

## GREEN evidence

Targeted and ASR regression suite:

```text
python3 -m unittest tests.test_stable_prefix tests.test_faster_whisper_streaming \
  tests.test_asr_pipeline tests.test_asr_provider -v
21 tests: OK
```

Benchmark model-free invocation:

```text
python3 scripts/benchmark_streaming_asr.py --help
exit 0
```

Full repository regression:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -q
Ran 310 tests: OK
```

## Coverage added

- Stable-prefix common-prefix example, correction/shrink, empty hypothesis,
  Unicode, finalization-once, and reset.
- Exact 200 ms and 400 ms cadence boundaries, deterministic rolling truncation,
  forced final decode below cadence, PCM metadata validation, and final delta
  accumulation.
- Sync/async injected runtimes, faster-whisper tuple/segment normalization,
  event-loop responsiveness, concurrent push/finalize serialization, and
  cancellation suppression of late output.
- `TranscriptChunk` positional compatibility and revision validation.
- Benchmark `--help` without model loading.

## Commit

`feat: add rolling faster whisper asr`

## Concerns / follow-up

No hardware model benchmark was run in this task. The benchmark script records
decode durations and RTF for the supplied local faster-whisper model; real GPU
latency, language options, and context sizing should be measured before Task
13 wires this provider into the live server.

---

## Fix round 1: review hardening

### RED evidence

The review regressions were added before the fix and executed with:

```text
python3 -m unittest tests.test_faster_whisper_streaming -v
Ran 15 tests: 3 failures, 4 errors
```

The failures demonstrated all accepted findings: no authoritative final
replacement fields, a lazy synchronous segments iterator blocking the event
loop, no project-path bootstrap for an absolute benchmark invocation, reused
revision IDs after reset, no shifted-window reconstruction, and no lifetime
RTF metrics.

### Changes

- [src/asr/stream.py](../../../src/asr/stream.py) adds optional,
  backward-compatible `committed_text` and `replaces_committed` fields after
  the existing positional fields. A final correction can now instruct
  revision-aware consumers to replace all formerly committed text exactly.
- [src/adapters/asr/providers/faster_whisper_streaming.py](../../../src/adapters/asr/providers/faster_whisper_streaming.py)
  consumes synchronous faster-whisper segment generators and normalizes their
  text inside `asyncio.to_thread`; it preserves a provider-lifetime publication
  sequence, stitches shifted rolling-window hypotheses through maximal
  suffix/prefix overlap, and exposes total decode/source/RTF metrics.
- [scripts/benchmark_streaming_asr.py](../../../scripts/benchmark_streaming_asr.py)
  now uses the repository-root bootstrap used by the other project scripts and
  prints both last-window and end-to-end metric labels.
- [tests/test_faster_whisper_streaming.py](../../../tests/test_faster_whisper_streaming.py)
  applies chunks through a small reference revision consumer for major final
  correction and shrink; it also covers the lazy generator, external CWD CLI,
  monotonic publications, shifted windows, and metric calculation.

The shifted-window stitch is intentionally limited to audio contexts that have
actually rolled beyond the configured buffer; normal unshifted decoder
corrections remain ordinary hypotheses and are not stitched.

### GREEN evidence

```text
python3 -m unittest tests.test_stable_prefix tests.test_faster_whisper_streaming \
  tests.test_asr_pipeline tests.test_asr_provider -v
Ran 28 tests: OK

python3 scripts/benchmark_streaming_asr.py --help
exit 0

# From an external temporary CWD, invoke the absolute script with missing
# paths: reaches "model directory does not exist" and never ModuleNotFoundError.

python3 -m unittest discover -s tests -p 'test_*.py' -q
Ran 317 tests: OK
```

### Commit

`fix: harden rolling asr revisions`

### Concerns

The overlap stitch is a bounded heuristic for window-local ASR text. It is
appropriate for V1 until timestamps or token-level alignment are available,
but real long-speech benchmarks should log overlap lengths and correction rates
before relying on it for production simultaneous interpretation.
