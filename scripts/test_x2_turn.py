"""Run the official local Transformers X2-Turn inference API."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main(model_path: str, audio_path: str, x2_root: str | None) -> None:
    if x2_root:
        source_root = Path(x2_root) / "src"
        sys.path.insert(0, str(source_root))

    try:
        import torch
        from transformers import AutoProcessor
        from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint
    except ImportError as exc:
        raise SystemExit(
            "X2-Turn runtime is unavailable. Install the official "
            "voxtral-realtime package and its transformers dependencies, "
            "or pass --x2-root pointing to its source directory."
        ) from exc

    print("loading X2-Turn...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = load_mtp_checkpoint(
        model_path,
        device="cuda",
        dtype=torch.bfloat16,
    ).eval()
    print("running ASR + turn inference...", flush=True)
    result = infer_asr_turn(model, processor, audio_path)

    print(f"transcript={result.transcript}")
    labels = Counter(frame.label for frame in result.turn_frames)
    print(f"turn_frames={len(result.turn_frames)}")
    print(f"turn_labels={dict(labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--x2-root")
    args = parser.parse_args()
    main(args.model, args.audio, args.x2_root)
