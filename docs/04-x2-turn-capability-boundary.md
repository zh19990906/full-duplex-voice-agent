# X2-Turn Integration Boundary

## Purpose

X2-Turn is treated as an optional realtime speech understanding and turn management component.

## We Use

- Streaming speech recognition
- Turn state prediction
- Backchannel detection
- End of turn prediction

## We Do Not Use X2-Turn For

- Dialogue reasoning
- User intent planning
- Long term memory
- Task execution
- Translation orchestration
- Text generation
- Speech synthesis

## Integration Rule

X2-Turn outputs events. Conversation Controller decides actions.

Example:

```
BACKCHANNEL -> continue generation
INTERRUPT -> stop output and revise
TURN_END -> accept user turn
```

Do not couple business logic directly to X2-Turn internals.
