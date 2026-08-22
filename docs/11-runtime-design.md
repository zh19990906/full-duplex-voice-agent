# Runtime Design

## Purpose

Define how multiple realtime models run together as one full duplex agent.

The runtime layer is responsible for orchestration, not model intelligence.

---

## Design Principle

Models are workers.

Runtime is the operating system layer.

```
Audio
 |
Runtime
 |
+---------+---------+---------+
|         |         |         |
Turn     LLM       TTS    Memory
Worker   Worker    Worker  Worker
```

---

## Model Lifecycle

Every model should support:

- load
- warmup
- inference
- unload
- health check

Example:

```
ModelManager
 |
 +-- ASR Adapter
 +-- Turn Adapter
 +-- LLM Adapter
 +-- TTS Adapter
```

---

## Realtime Scheduling

Priority:

1. User interruption handling
2. Audio output control
3. Turn detection
4. Generation
5. Background memory operations

User interaction latency has higher priority than throughput.

---

## GPU Resource Management

Future support:

- multiple models on one GPU
- model lazy loading
- quantized inference
- device placement configuration

Configuration should not be hard coded.

---

## Parallel Development Boundary

Suggested ownership:

```
Agent Runtime
Conversation Controller
Model Adapters
Benchmarks
Deployment
```

Each module should expose stable interfaces.
