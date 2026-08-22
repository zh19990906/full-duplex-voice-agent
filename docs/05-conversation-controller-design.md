# Conversation Controller Design

## Role

Conversation Controller is the realtime operating system layer of this project.

It coordinates multiple models and decides what action should happen.

## State Machine

```
IDLE
 |
LISTENING
 |
THINKING
 |
SPEAKING
 |
+----------------+
|                |
BACKCHANNEL   INTERRUPT
|                |
CONTINUE     REVISE
```

## Events

### BACKCHANNEL

Examples:

- yes
- ok
- continue

Action:

```
KEEP_GENERATING
KEEP_TTS
```

### INTERRUPT

Action:

```
STOP_TTS
CANCEL_GENERATION
UPDATE_CONTEXT
```

### RESUME

Action:

```
LOAD_TASK_STATE
CONTINUE_TASK
```

## Design Rules

1. Models never directly control conversation flow.
2. All realtime decisions go through Controller.
3. Business logic stays outside models.
4. Every state transition must be testable.
