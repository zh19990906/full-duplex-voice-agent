#!/usr/bin/env python3
"""Benchmark rolling faster-whisper ASR against a local PCM WAV file."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import wave


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _cadence(value: str) -> int:
    parsed = int(value)
    if not 200 <= parsed <= 400:
        raise argparse.ArgumentTypeError("cadence must be between 200 and 400 ms")
    return parsed


def _context_seconds(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 10:
        raise argparse.ArgumentTypeError("context must be greater than zero and at most 10 seconds")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local faster-whisper model directory")
    parser.add_argument("--audio", required=True, help="16kHz mono PCM16 WAV input")
    parser.add_argument("--cadence-ms", type=_cadence, default=300)
    parser.add_argument("--context-seconds", type=_context_seconds, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language")
    return parser


async def _run(args: argparse.Namespace) -> None:
    from src.adapters.asr.providers.faster_whisper_streaming import (
        FasterWhisperStreamingProvider,
    )
    from src.realtime.protocol import PCM16_FRAME_BYTES

    audio_path = Path(args.audio)
    if not Path(args.model).is_dir():
        raise ValueError(f"model directory does not exist: {args.model}")
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    with wave.open(str(audio_path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("audio must be 16kHz, mono, PCM16 WAV")
        pcm = source.readframes(source.getnframes())

    provider = FasterWhisperStreamingProvider(
        args.model,
        load_model=True,
        cadence_ms=args.cadence_ms,
        context_seconds=args.context_seconds,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    for start in range(0, len(pcm), PCM16_FRAME_BYTES):
        frame = pcm[start : start + PCM16_FRAME_BYTES]
        if len(frame) != PCM16_FRAME_BYTES:
            break
        chunk = await provider.push_pcm(frame)
        if chunk is not None:
            print(chunk.to_dict())
    print((await provider.finalize_turn()).to_dict())
    print(
        {
            "decode_durations": provider.decode_durations,
            "last_window_rtf": provider.last_rtf,
            "total_decode_seconds": provider.total_decode_seconds,
            "source_audio_seconds": provider.source_audio_seconds,
            "end_to_end_decode_rtf": provider.end_to_end_decode_rtf,
        }
    )


def main() -> None:
    args = _parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
