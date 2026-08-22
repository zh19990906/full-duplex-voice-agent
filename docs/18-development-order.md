# Development Order

## Principle

Build the realtime interaction loop first, then extend capabilities.

## Phase 1: Foundation

1. Define event protocol
2. Implement audio streaming interface
3. Implement model adapters

## Phase 2: Full Duplex Core

1. Conversation Controller
2. X2-Turn integration
3. Generation Manager
4. Streaming TTS

Target:

- user backchannel continues response
- user interruption stops response

## Phase 3: Stateful Capability

1. Task State Manager
2. Resume workflow
3. Persistent session state

## Phase 4: Translation

1. Streaming ASR
2. Incremental translation
3. Streaming TTS

## Phase 5: Optimization

1. Runtime scheduler
2. Latency optimization
3. Resource management

## Parallel Development Rule

Codex agents may work in parallel only when module interfaces are stable.
Shared protocol changes require review.
