"""Standalone CosyVoice3 JSONL worker.

Run this script with the CosyVoice virtual environment. The main application
communicates with it over stdin/stdout and receives mono PCM16 audio chunks.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


class CosyVoiceWorker:
    def __init__(self, model_path: str) -> None:
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self._torch = torch
        self._model = AutoModel(model_dir=model_path)
        self._sample_rate = int(self._model.sample_rate)
        self._cancel = threading.Event()
        self._output_lock = threading.Lock()
        self._active: threading.Thread | None = None
        self._protocol_stdout = sys.stdout

    def run(self) -> None:
        self._write({"type": "ready", "sample_rate": self._sample_rate})
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            try:
                request = json.loads(raw_line)
                operation = request.get("op")
                if operation == "synthesize":
                    self._start_synthesis(request)
                elif operation == "cancel":
                    self._cancel.set()
                elif operation == "shutdown":
                    self._cancel.set()
                    return
                else:
                    self._write({"type": "error", "error": f"unknown operation: {operation!r}"})
            except Exception as exc:  # pragma: no cover - defensive process boundary
                self._write({"type": "error", "error": str(exc)})

    def _start_synthesis(self, request: dict[str, Any]) -> None:
        if self._active is not None and self._active.is_alive():
            self._write(
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "error": "CosyVoice worker is already synthesizing",
                }
            )
            return
        self._cancel.clear()
        self._active = threading.Thread(
            target=self._synthesize,
            args=(request,),
            daemon=True,
        )
        self._active.start()

    def _synthesize(self, request: dict[str, Any]) -> None:
        request_id = str(request["request_id"])
        try:
            prompt_audio = request.get("prompt_audio")
            prompt_text = request.get("prompt_text")
            if not prompt_audio or not prompt_text:
                raise ValueError("prompt_audio and prompt_text are required")
            with redirect_stdout(sys.stderr):
                chunks = self._model.inference_zero_shot(
                    str(request["text"]),
                    str(prompt_text),
                    str(prompt_audio),
                    stream=True,
                )
                for index, item in enumerate(chunks):
                    if self._cancel.is_set():
                        self._write({"type": "cancelled", "request_id": request_id})
                        return
                    pcm = self._to_pcm16(item["tts_speech"])
                    self._write(
                        {
                            "type": "chunk",
                            "request_id": request_id,
                            "chunk_id": f"cosyvoice-{request_id}-{index}",
                            "audio_b64": base64.b64encode(pcm).decode("ascii"),
                            "timestamp": time.time(),
                            "is_final": False,
                        }
                    )
            self._write(
                {
                    "type": "chunk",
                    "request_id": request_id,
                    "chunk_id": f"cosyvoice-{request_id}-final",
                    "audio_b64": "",
                    "timestamp": time.time(),
                    "is_final": True,
                }
            )
            self._write({"type": "done", "request_id": request_id})
        except Exception as exc:  # pragma: no cover - exercised in real env
            self._write(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )

    def _to_pcm16(self, speech: Any) -> bytes:
        tensor = speech.detach().float().cpu()
        tensor = tensor.reshape(-1).clamp(-1.0, 1.0)
        return (tensor * 32767.0).short().numpy().tobytes()

    def _write(self, message: dict[str, Any]) -> None:
        with self._output_lock:
            self._protocol_stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._protocol_stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated CosyVoice3 JSONL worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cosy-root", default=None)
    args = parser.parse_args()
    if args.cosy_root:
        root = Path(args.cosy_root).resolve()
        sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
        sys.path.insert(0, str(root))
    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        worker = CosyVoiceWorker(args.model)
    worker._protocol_stdout = protocol_stdout
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
