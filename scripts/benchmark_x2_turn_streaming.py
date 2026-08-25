#!/usr/bin/env python3
"""Benchmark local X2-Turn rolling inference on a 16kHz mono PCM WAV file."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import tempfile
import wave


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _cadence(value: str) -> int:
    parsed = int(value)
    if not 100 <= parsed <= 200:
        raise argparse.ArgumentTypeError("cadence must be between 100 and 200 ms")
    return parsed


def _context_seconds(value: str) -> float:
    parsed = float(value)
    if not 1.0 <= parsed <= 3.0:
        raise argparse.ArgumentTypeError("context must be between 1 and 3 seconds")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local X2-Turn model directory")
    parser.add_argument("--audio", required=True, help="16kHz mono PCM16 WAV input")
    parser.add_argument("--x2-root", help="official X2-Turn repository root")
    parser.add_argument("--cadence-ms", type=_cadence, default=160)
    parser.add_argument("--context-seconds", type=_context_seconds, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser


class _OfficialX2Runtime:
    """Thin, lazy official-runtime bridge used only by the hardware benchmark."""

    def __init__(self, model_path: str, x2_root: str | None, device: str) -> None:
        if x2_root:
            source_root = Path(x2_root) / "src"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
        try:
            import torch
            from transformers import AutoProcessor
            from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                "official X2-Turn runtime is unavailable; install voxtral-realtime "
                "or pass --x2-root"
            ) from exc
        self._infer_asr_turn = infer_asr_turn
        self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self._model = load_mtp_checkpoint(
            model_path, device=device, dtype=torch.bfloat16
        ).eval()

    def infer(self, pcm: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(pcm)
            return self._infer_asr_turn(self._model, self._processor, temporary.name)


async def _run(args: argparse.Namespace) -> None:
    from src.adapters.turn.x2_turn_streaming import X2TurnRollingProvider
    from src.realtime.protocol import PCM16_FRAME_BYTES

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    with wave.open(str(audio_path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("audio must be 16kHz, mono, PCM16 WAV")
        pcm = source.readframes(source.getnframes())

    runtime = _OfficialX2Runtime(str(model_path), args.x2_root, args.device)
    provider = X2TurnRollingProvider(
        str(model_path),
        runtime=runtime,
        cadence_ms=args.cadence_ms,
        context_seconds=args.context_seconds,
    )
    for start in range(0, len(pcm), PCM16_FRAME_BYTES):
        frame = pcm[start : start + PCM16_FRAME_BYTES]
        if len(frame) != PCM16_FRAME_BYTES:
            break
        for candidate in await provider.push_pcm(frame):
            print(candidate.to_dict())
    for candidate in await provider.finalize_turn():
        print(candidate.to_dict())
    print(
        {
            "inference_durations": provider.decode_durations,
            "last_window_rtf": provider.last_rtf,
            "total_inference_seconds": provider.total_decode_seconds,
            "source_audio_seconds": provider.source_audio_seconds,
            "total_rtf": provider.end_to_end_decode_rtf,
        }
    )


def main() -> None:
    args = _parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
