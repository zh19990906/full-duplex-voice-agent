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
