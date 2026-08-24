# Full Duplex Voice Agent

A GPT Live-like full duplex voice agent architecture based on modular multimodal models and realtime orchestration.

## Goal

This project focuses on four core capabilities:

1. Backchannel continuation: users can say "嗯嗯/对" without interrupting assistant speech.
2. Interruption and revision: users can interrupt and change the direction of the answer.
3. Simultaneous translation: realtime speech translation with streaming output.
4. Task state recovery: interrupted tasks can resume from a saved state.

## Design Philosophy

The first stage avoids modifying foundation model architectures. Instead we combine:

- Streaming ASR
- Turn management
- Conversation controller
- Stateful LLM
- Task memory
- Streaming TTS

The main engineering value is the realtime orchestration layer.

## Real local model stack

The Qwen and X2-Turn models can run in the main project environment. CosyVoice3
is intentionally isolated in `/home/CosyVoice/.venv` because it has its own
PyTorch/CUDA dependencies. The JSONL worker keeps CosyVoice loaded and returns
mono PCM16 chunks to the main process.

From the main project environment, run the worker smoke test:

```bash
cd /code/full-duplex-voice-agent
source .venv/bin/activate
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib
unset LD_PRELOAD

python scripts/test_cosyvoice_worker.py \
  --model /mnt/model-zhangheng/Fun-CosyVoice3-0.5B-2512 \
  --prompt-audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --prompt-text 'hello can you tell me what the weather is like today<|endofprompt|>' \
  --text '你好，这是一次独立 CosyVoice worker 的流式语音合成测试。' \
  --output /tmp/cosyvoice_worker_test.wav
```

The main application can construct the provider with `provider:
cosyvoice_worker` and options for `worker_python`, `prompt_audio`, and
`prompt_text`; the worker script itself must be launched with the CosyVoice
virtual environment.

## High Level Architecture

```
Audio
 |
 v
Streaming Speech Layer
 |
 +--> ASR
 +--> Turn Manager
          |
          v
Conversation Controller
          |
 +--------+---------+
 |                  |
LLM             Task Memory
 |
 v
Streaming TTS
```

## Development Principle

Models are replaceable components. Business logic belongs in controllers, state managers and adapters.
