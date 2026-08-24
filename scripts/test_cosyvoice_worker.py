"""Smoke-test the isolated CosyVoice worker from the main project environment."""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.tts.providers.cosyvoice_worker import CosyVoiceWorkerClient


async def run(args: argparse.Namespace) -> None:
    client = CosyVoiceWorkerClient(
        args.model,
        worker_python=args.worker_python,
        worker_script=args.worker_script,
        cosyvoice_root=args.cosy_root,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = await client.stream_audio(args.text)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(args.sample_rate)
            async for chunk in result:
                wav.writeframes(chunk.audio_data)
                if chunk.is_final:
                    break
        print(f"saved={output}")
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a CosyVoice JSONL worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="/tmp/cosyvoice_worker_test.wav")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--worker-python", default="/home/CosyVoice/.venv/bin/python")
    parser.add_argument(
        "--worker-script",
        default=str(Path(__file__).with_name("cosyvoice_worker.py")),
    )
    parser.add_argument("--cosy-root", default="/home/CosyVoice")
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
