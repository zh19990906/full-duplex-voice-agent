# Model Adapter Interface Design

## Goal

Prevent the system from coupling to a specific model implementation.

All AI models should be accessed through adapters.

---

## Turn Adapter

Responsibilities:

- receive audio stream
- detect turn state
- detect interruption

Example events:

```
BACKCHANNEL
INTERRUPT
TURN_END
UNKNOWN
```

Possible backend:

- X2-Turn
- other turn detection models

---

## ASR Adapter

Interface:

```
stream_audio(chunk)
 -> partial transcript
```

Requirements:

- streaming output
- timestamps
- confidence

---

## LLM Adapter

Responsibilities:

- prompt management
- streaming generation
- cancellation

Required operations:

```
generate()
stream_tokens()
cancel()
```

---

## TTS Adapter

Responsibilities:

- text streaming
- audio streaming
- immediate stop

Required operations:

```
speak()
stream_audio()
interrupt()
```

---

## Memory Adapter

Responsibilities:

- conversation state
- task state
- long-running workflow state

---

## Adapter Rules

1. Business logic must not exist inside adapters.
2. Adapters only translate runtime calls to model APIs.
3. New models should be added by creating new adapters.
