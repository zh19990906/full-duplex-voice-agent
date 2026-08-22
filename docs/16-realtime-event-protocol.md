# Realtime Event Protocol

## Purpose

Define the common event language between realtime modules. All realtime decisions must be expressed through events instead of direct model-to-model calls.

## Core Events

```text
USER_SPEECH_START
USER_SPEECH_PARTIAL
USER_BACKCHANNEL
USER_INTERRUPT
USER_TURN_END

ASSISTANT_SPEECH_START
ASSISTANT_SPEECH_CHUNK
ASSISTANT_SPEECH_STOP

TASK_PAUSE
TASK_RESUME
TASK_STATE_UPDATE
```

## Event Example

```json
{
  "event": "USER_INTERRUPT",
  "timestamp": 123456,
  "confidence": 0.95,
  "transcript": "等等，我想问上海"
}
```

## Design Rules

1. Models emit events, Controller decides actions.
2. High priority realtime events can preempt generation.
3. Event schema must remain stable across model replacements.
4. All benchmarks should map to event sequences.
