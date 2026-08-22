# Task State Memory Design

## Motivation

Long-running tasks cannot depend only on LLM conversation history.

Example:

Assistant:
```
1 2 3 4 5
```

User:
```
Stop for a moment
```

User:
```
Continue
```

The system must continue from 6.

## Design

Introduce explicit task state.

```
Task State Manager
        |
        v
Persistent State
```

Example:

```json
{
  "task": "counting",
  "current_value": 5,
  "step": 1,
  "direction": "increase"
}
```

## Responsibilities

Task State Manager handles:

- task creation
- checkpointing
- interruption recovery
- resume execution

## Supported Future Scenarios

- counting
- reading long documents
- multi-step agent workflows
- tool execution recovery
- long conversations

## Principle

LLM generates language. Task State Manager maintains execution state.

Do not force the LLM to remember operational state.
