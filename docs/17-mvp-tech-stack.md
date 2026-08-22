# MVP Technology Stack

## Goal

Freeze the first implementation choices to avoid endless technology selection.

## Proposed Stack

| Component | Choice | Reason |
|---|---|---|
| Turn Management | X2-Turn adapter | realtime turn detection |
| Streaming ASR | X2-Turn compatible streaming ASR | reduce latency |
| LLM | API or local LLM adapter | replaceable reasoning layer |
| TTS | Streaming TTS adapter | realtime response |
| Runtime | Python asyncio | fast iteration |
| Transport | WebSocket | bidirectional streaming |
| Memory | Redis / KV store | task state persistence |

## MVP Capability Target

Must support:

- Backchannel continuation
- Basic interruption
- Response cancellation
- Streaming audio loop

Not included:

- End-to-end speech model training
- Advanced multimodal input
- Large scale distributed serving

## Model Principle

Models are replaceable modules. Business logic belongs to Controller, not models.
