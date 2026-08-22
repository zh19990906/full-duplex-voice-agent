# Architecture Decision Records

This document records important architectural decisions for the full-duplex voice agent project.

The goal is to keep future Codex agents and developers aligned when modifying the system.

---

# ADR-001: Use modular multi-model architecture first

## Status
Accepted

## Decision

The first implementation will use multiple specialized models connected by an orchestration layer instead of training an end-to-end speech-to-speech model.

Target architecture:

```
Streaming Speech
        |
        v
Turn / ASR Layer
        |
        v
Conversation Controller
        |
 +------+------+
 |             |
LLM        Task Memory
 |
 v
Streaming TTS
```

## Reason

The initial target is GPT Live-like interaction capabilities, not foundation model training.

A modular architecture enables:

- faster iteration
- model replacement
- offline deployment
- easier debugging
- parallel Codex development

---

# ADR-002: X2-Turn is a capability component, not the whole system

## Status
Accepted

## Decision

X2-Turn is integrated as a turn-management capability.

Used for:

- streaming speech understanding
- turn state prediction
- backchannel detection
- end-of-turn prediction

Not responsible for:

- reasoning
- response generation
- memory
- translation planning
- task execution
- TTS

## Reason

Full duplex capability is an orchestration problem, not only a turn detection problem.

---

# ADR-003: Conversation Controller is the central control plane

## Status
Accepted

## Decision

All realtime interaction decisions must go through Conversation Controller.

Examples:

- continue speaking after backchannel
- interrupt TTS
- cancel generation
- update user intent
- resume task

Models should not directly control system behavior.

---

# ADR-004: Task state should not depend only on LLM context

## Status
Accepted

## Decision

Long-running tasks require explicit state storage.

Example:

```json
{
  "task": "counting",
  "current_value": 10,
  "step": 1
}
```

The LLM can reason over task state, but the state itself belongs to Task State Manager.

## Reason

This improves reliability for:

- counting
- workflows
- tool execution
- long conversations

---

# ADR-005: Every model must be accessed through adapters

## Status
Accepted

## Decision

Models must expose stable interfaces.

Examples:

- TurnAdapter
- ASRAdapter
- LLMAdapter
- TTSAdapter
- TranslationAdapter

Replacing a model should not require changing business logic.
