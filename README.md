# Full Duplex Voice Agent

A GPT Live-like full duplex voice agent architecture based on modular multimodal models and realtime orchestration.

## Goal

This project focuses on four core capabilities:

1. Backchannel continuation: users can say "嗯嗯/对" without interrupting assistant speech.
2. Interruption and revision: users can interrupt and change the direction of the answer.
3. Simultaneous translation: realtime speech translation with streaming output.
4. Task state recovery: interrupted tasks can resume from a saved state.

## Design Philosophy

The first stage avoids modifying foundation model architectures. Instead we combine:

- Streaming ASR
- Turn management
- Conversation controller
- Stateful LLM
- Task memory
- Streaming TTS

The main engineering value is the realtime orchestration layer.

## High Level Architecture

```
Audio
 |
 v
Streaming Speech Layer
 |
 +--> ASR
 +--> Turn Manager
          |
          v
Conversation Controller
          |
 +--------+---------+
 |                  |
LLM             Task Memory
 |
 v
Streaming TTS
```

## Development Principle

Models are replaceable components. Business logic belongs in controllers, state managers and adapters.
