# Model Stack Selection

## Overview

The project aims to implement GPT Live-like full duplex interaction capability through modular components. The first stage does not modify foundation model architectures. Instead, we combine existing models with a realtime orchestration layer.

## Model Layers

| Layer | Responsibility | Initial Strategy |
|---|---|---|
| Turn Management | Detect user interaction behavior | X2-Turn or equivalent |
| Streaming ASR | Convert audio stream into text | Reuse existing models |
| LLM | Reasoning and response generation | External or local LLM |
| Translation | Streaming language conversion | LLM/NMT based |
| TTS | Streaming speech generation | Existing realtime TTS |

## Design Principle

Models are replaceable modules. No business logic should be embedded inside model wrappers.

```
Audio
  |
  v
Model Adapter
  |
  v
Conversation Controller
  |
  v
Agent Runtime
```

## Recommended First Version

- Turn: X2-Turn
- ASR: streaming ASR backend
- Reasoning: GPT/Qwen/Llama compatible interface
- TTS: streaming TTS backend
- Memory: external state storage

## Training Strategy

Phase 1:
- No model training
- Focus on orchestration

Phase 2:
- Collect interruption/backchannel data
- Improve turn prediction

Phase 3:
- Consider end-to-end speech-to-speech models
