# Codex Project Rules

## Architecture Rules

1. Models must be replaceable through adapters.
2. Realtime decisions must go through Conversation Controller.
3. Do not put business logic inside model wrappers.
4. Every feature requires benchmark scenarios.

## Parallel Development

Modules should have clear ownership:

- turn_manager
- conversation_controller
- model_adapter
- memory_manager
- benchmark

Avoid changing shared interfaces without documenting an Architecture Decision Record.

## Implementation Priority

1. Make the system observable.
2. Add tests for the four target cases.
3. Optimize latency after correctness.
