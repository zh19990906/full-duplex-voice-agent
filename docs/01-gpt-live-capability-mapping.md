# GPT Live Capability Mapping

## Objective

This project does not attempt to reproduce OpenAI GPT Live internals. The objective is to reproduce several key user experiences through a modular real-time architecture.

## Target Capabilities

### 1. Backchannel Continuation

User signals:

- 嗯嗯
- 对
- 好的
- 继续

Expected behavior:

- Do not stop TTS
- Do not reset generation
- Continue current response

Required modules:

- Turn detector
- Conversation controller

---

### 2. Interruption and Intent Revision

User can interrupt:

Example:

Assistant: 北京旅游有几个建议...

User: 等一下，我想问上海

Expected behavior:

1. Stop current audio
2. Cancel current generation
3. Update user intent
4. Regenerate response

Required modules:

- Turn detector
- Generation manager
- Dialogue state manager

---

### 3. Streaming Translation

Goal:

Speech input -> translated speech output with low latency.

Pipeline:

Audio -> Streaming ASR -> Translation -> Streaming TTS

---

### 4. Stateful Task Resume

Example:

Assistant counts numbers.

User interrupts.

System stores:

```
task=counting
current_number=N
```

Resume from state instead of relying only on conversation history.
