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

Run the complete offline backend chain without starting the browser frontend:

```bash
python scripts/test_real_stack.py \
  --asr-model /mnt/model-zhangheng/X2-Turn-4B-0812 \
  --llm-model /mnt/model-zhangheng/Qwen2.5-14B-Instruct \
  --tts-model /mnt/model-zhangheng/Fun-CosyVoice3-0.5B-2512 \
  --audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --x2-root /home/X2-Turn/voxtral-realtime \
  --prompt-audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --prompt-text 'hello can you tell me what the weather is like today<|endofprompt|>' \
  --output /tmp/real_stack_response.wav
```

To serve the real text-to-speech path and the browser console on the mapped
port `8001`:

```bash
python -m pip install fastapi uvicorn

python scripts/run_real_server.py \
  --llm-model /mnt/model-zhangheng/Qwen2.5-14B-Instruct \
  --tts-model /mnt/model-zhangheng/Fun-CosyVoice3-0.5B-2512 \
  --prompt-audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --prompt-text 'hello can you tell me what the weather is like today<|endofprompt|>' \
  --port 8001
```

Open `http://服务器地址:8001`. This browser path currently supports text input
through the real Qwen and CosyVoice models. Binary microphone messages return a
clear notice because the checked-in X2-Turn local API performs offline whole-file
inference rather than streaming raw browser audio.

## Realtime acceptance

Use the acceptance CLI to evaluate a report with an explicit evidence label:

```bash
python3 scripts/run_realtime_acceptance.py --help
```

Accepted labels are `unit`, `simulated`, `model-integration`, and
`hardware-e2e`. The CLI refuses to pass a `hardware-e2e` report unless it is
explicitly marked as a real run and all required measurements are present.

Recorded-audio model integration on the GPU server:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label model-integration \
  --audio /home/X2-Turn/turn-demo/assets/sample_en.wav \
  --report /tmp/realtime-model-integration.json
```

Real browser/headset acceptance on the target machine:

```bash
python3 scripts/run_realtime_acceptance.py \
  --profile local_gpu \
  --label hardware-e2e \
  --explicit-real-run \
  --report /tmp/realtime-hardware-e2e.json
```

Required V1 hard gates:

- `duck_latency_ms <= 100`
- `interrupt_latency_ms <= 250`
- `backchannel_restore_latency_ms <= 300`
- `first_token_latency_ms <= 800`
- `first_audio_latency_ms <= 1500`
- `first_translated_audio_latency_ms <= 2000`
- `stale_output_count == 0`
- `resume_phrase_error_count <= 1`

GPU, headset, and 60-minute soak validation are external gates. Run them on the
target deployment and record them separately; do not relabel synthetic or
partial evidence as `hardware-e2e`.

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
