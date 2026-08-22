# Interruption and Intent Revision Design

## Goal

Support the core GPT Live behavior:

User can interrupt the assistant and change the direction of the answer.

Example:

Assistant:
> Beijing travel has three major attractions...

User:
> Wait, I want Shanghai instead.

Expected behavior:

1. Stop current speech output
2. Cancel current generation
3. Understand new intent
4. Regenerate response

## Pipeline

```
User Audio
   |
   v
Turn Detector
   |
   +--> INTERRUPT
             |
             v
       Generation Manager
             |
       Stop TTS
       Cancel decoding
             |
             v
       Context Revision
             |
             v
          LLM
```

## Components

### Turn Manager

Responsible for detecting:

- backchannel
- interruption
- turn completion

### Generation Manager

Responsible for:

- streaming token cancellation
- TTS interruption
- generation lifecycle

### Context Revision

The system should not simply append new text. It should update the current conversation goal.

Example:

Before:

```
intent = Beijing travel
```

After:

```
intent = Shanghai travel
```

## Future Optimization

Potential improvements:

- KV cache checkpointing
- speculative interruption prediction
- partial intent recognition

## Non-goal

The first version does not require modifying the LLM architecture.
