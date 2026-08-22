# Streaming Translation Design

## Goal

Implement GPT Live-like low latency speech translation without requiring an end-to-end speech-to-speech model in the first phase.

Target capability:

- User speaks in language A
- System understands partial speech
- System produces translated speech in language B with low latency

---

## Architecture

```
Audio Stream
    |
    v
Streaming ASR
    |
    v
Incremental Translation
    |
    v
Streaming TTS
    |
    v
Translated Audio
```

---

## Components

### Streaming ASR

Responsibilities:

- audio chunk processing
- partial transcript generation
- timestamp alignment

Possible implementations:

- X2-Turn ASR capability
- Whisper streaming variants
- FunASR streaming

---

### Translation Engine

Responsibilities:

- incremental translation
- context preservation
- translation stability

The first version can use:

- LLM based translation
- NMT models

The translation layer must be replaceable through adapters.

---

### Streaming TTS

Responsibilities:

- consume partial translated text
- generate audio chunks
- support interruption

---

## Latency Targets

The system should optimize:

- time to first translated audio
- translation stability
- correction handling

Initial target:

```
Speech input -> translated speech output
< 1-2 seconds
```

---

## Interaction With Conversation Controller

Translation mode is still controlled by the same runtime:

```
Audio
 |
Turn Manager
 |
Conversation Controller
 |
Translation Workflow
```

The translation workflow must support:

- interruption
- pause
- resume
- language switching
