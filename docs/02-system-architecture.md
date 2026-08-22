# System Architecture

## Design Principle

The system is built as a real-time orchestration system instead of a single end-to-end model.

```
Audio Input
    |
    v
+----------------+
| Speech Layer   |
| ASR + Turn     |
+----------------+
    |
    v
+----------------------+
| Conversation         |
| Controller           |
+----------------------+
    |
 +--+---------+---------+
 |            |         |
 v            v         v
LLM       Task Memory Translation
 |
 v
Streaming TTS
 |
 v
Audio Output
```

## Components

### Speech Layer

Responsibilities:

- streaming speech recognition
- turn state prediction
- audio event detection

Possible backend:

- X2-Turn
- other streaming speech models

### Conversation Controller

The central component.

Responsibilities:

- decide continue / interrupt / revise / resume
- coordinate model execution
- maintain realtime state

### Generation Manager

Responsibilities:

- token cancellation
- streaming generation control
- response replacement

### Task Memory

Stores executable task state.

Example:

```json
{
  "task": "count",
  "current": 10
}
```

## Non-goals

First phase does not modify foundation model architecture.

Models should remain replaceable through adapters.
