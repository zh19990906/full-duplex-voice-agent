"""Run the real offline ASR -> LLM -> TTS backend chain."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def run_asr(model_path: str, audio_path: str, x2_root: str | None):
    if x2_root:
        sys.path.insert(0, str(Path(x2_root) / "src"))

    import torch
    from transformers import AutoProcessor
    from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint

    print("[1/3] loading X2-Turn...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = load_mtp_checkpoint(
        model_path,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    result = infer_asr_turn(model, processor, audio_path)
    labels = Counter(frame.label for frame in result.turn_frames)
    print(f"transcript={result.transcript}", flush=True)
    print(f"turn_frames={len(result.turn_frames)}", flush=True)
    print(f"turn_labels={dict(labels)}", flush=True)
    return result.transcript


async def run_llm(model_path: str, transcript: str) -> str:
    from src.adapters.llm.providers.qwen_transformers import TransformersQwenProvider

    print("[2/3] loading Qwen...", flush=True)
    provider = TransformersQwenProvider(model_path, device="cuda")
    prompt = (
        "请回答用户的问题，回答要自然、简洁，适合直接转换成语音。\n"
        f"用户：{transcript}\n"
        "助手："
    )
    pieces: list[str] = []
    async for chunk in provider.stream_tokens(prompt):
        pieces.append(chunk.text)
    response = "".join(pieces).strip()
    print(f"response={response}", flush=True)
    return response


async def run_tts(args: argparse.Namespace, text: str) -> None:
    from src.adapters.tts.providers.cosyvoice_worker import CosyVoiceWorkerClient

    print("[3/3] starting CosyVoice worker...", flush=True)
    client = CosyVoiceWorkerClient(
        args.tts_model,
        worker_python=args.worker_python,
        worker_script=args.worker_script,
        cosyvoice_root=args.cosy_root,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = await client.stream_audio(text)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(args.sample_rate)
            async for chunk in result:
                wav.writeframes(chunk.audio_data)
                if chunk.is_final:
                    break
    finally:
        await client.close()
    print(f"output={output}", flush=True)


async def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    transcript = run_asr(args.asr_model, args.audio, args.x2_root)
    if not transcript.strip():
        raise RuntimeError("X2-Turn returned an empty transcript")
    response = await run_llm(args.llm_model, transcript)
    if not response:
        raise RuntimeError("Qwen returned an empty response")
    await run_tts(args, response)
    print(f"total_seconds={time.perf_counter() - started:.3f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real offline ASR -> LLM -> TTS chain")
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--x2-root", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--output", default="/tmp/real_stack_response.wav")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--worker-python", default="/home/CosyVoice/.venv/bin/python")
    parser.add_argument("--cosy-root", default="/home/CosyVoice")
    parser.add_argument(
        "--worker-script",
        default=str(Path(__file__).with_name("cosyvoice_worker.py")),
    )
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
