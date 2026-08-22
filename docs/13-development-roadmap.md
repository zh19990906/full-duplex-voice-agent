# Development Roadmap

## Goal

Build a GPT Live-like full duplex voice agent through modular realtime components.

The project prioritizes engineering orchestration over training a new end-to-end foundation model.

---

# Phase 0: Foundation

Goal: establish architecture and development conventions.

Tasks:

- Repository structure
- Documentation
- Model configuration
- Adapter interfaces
- Codex development rules

Exit criteria:

- Architecture is stable
- Modules can be developed independently

---

# Phase 1: Realtime Full Duplex Core

Goal: achieve natural voice interaction.

Capabilities:

- Streaming ASR
- Turn detection
- Conversation Controller
- Streaming TTS

Target cases:

- User backchannel does not interrupt response
- User can interrupt assistant

Main components:

- X2-Turn Adapter
- Conversation Controller
- Generation Manager

---

# Phase 2: Task Continuity

Goal: support resumable tasks.

Capabilities:

- Task state storage
- Pause/resume lifecycle
- Long running interaction state

Examples:

- Counting
- Reading continuation
- Agent workflows

---

# Phase 3: Realtime Translation

Goal: support simultaneous interpretation.

Components:

- Streaming ASR
- Incremental translation
- Streaming TTS

Metrics:

- Translation quality
- First audio latency
- Stability

---

# Phase 4: Runtime Optimization

Goal: approach production quality.

Focus:

- Model scheduling
- GPU resource management
- Latency optimization
- Failure recovery

---

# Phase 5: Advanced Intelligence

Future direction:

- Emotion awareness
- Multimodal input
- Vision integration
- More human-like turn taking

---

# Development Principles

1. Keep models replaceable.
2. Put realtime decisions into controllers, not models.
3. Every capability requires benchmark coverage.
4. Prefer engineering composition before model modification.
