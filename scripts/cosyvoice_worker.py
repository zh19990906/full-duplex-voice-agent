"""Standalone, request-cancellable CosyVoice3 JSONL worker."""

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
    """Run exactly one cooperative CosyVoice synthesis request at a time."""

    def __init__(self, model_path: str) -> None:
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self._torch = torch
        self._model = AutoModel(model_dir=model_path)
        self._sample_rate = int(self._model.sample_rate)
        self._output_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: threading.Thread | None = None
        self._active_request_id: str | None = None
        self._active_cancel: threading.Event | None = None
        self._protocol_stdout = sys.stdout

    def run(self) -> None:
        self._write({"type": "ready", "sample_rate": self._sample_rate})
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            request: Any = None
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("worker command must be a JSON object")
                if self._dispatch(request):
                    return
            except Exception as exc:  # pragma: no cover - defensive process boundary
                request_id = request.get("request_id") if isinstance(request, dict) else None
                message: dict[str, Any] = {"type": "error", "error": str(exc)}
                if isinstance(request_id, str) and request_id:
                    message["request_id"] = request_id
                self._write(message)

    def _dispatch(self, request: dict[str, Any]) -> bool:
        operation = request.get("op")
        if operation == "synthesize":
            self._start_synthesis(request)
            return False
        if operation == "cancel":
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("cancel requires a nonempty request_id")
            self._cancel_request(request_id)
            return False
        if operation == "shutdown":
            self._shutdown()
            return True
        self._write({"type": "error", "error": f"unknown operation: {operation!r}"})
        return False

    def _start_synthesis(self, request: dict[str, Any]) -> None:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("synthesize requires a nonempty request_id")
        if not isinstance(request.get("text"), str):
            raise ValueError("synthesize requires text")
        with self._state_lock:
            if self._active is not None and self._active.is_alive():
                self._write(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": "CosyVoice worker is already synthesizing",
                    }
                )
                return
            cancel = threading.Event()
            active = threading.Thread(target=self._synthesize, args=(request, cancel), daemon=True)
            self._active = active
            self._active_request_id = request_id
            self._active_cancel = cancel
            active.start()

    def _cancel_request(self, request_id: str) -> None:
        with self._state_lock:
            if self._active_request_id != request_id or self._active_cancel is None:
                self._write(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": "no matching active CosyVoice request",
                    }
                )
                return
            self._active_cancel.set()

    def _shutdown(self) -> None:
        with self._state_lock:
            active = self._active
            cancel = self._active_cancel
            if cancel is not None:
                cancel.set()
        if active is not None and active.is_alive():
            active.join(timeout=5)

    def _synthesize(self, request: dict[str, Any], cancel: threading.Event) -> None:
        request_id = str(request["request_id"])
        terminal_sent = False
        identity = {
            name: request[name]
            for name in ("response_id", "generation_epoch", "segment_id")
            if name in request
        }

        def cancelled() -> bool:
            nonlocal terminal_sent
            if not cancel.is_set():
                return False
            if not terminal_sent:
                self._write({"type": "cancelled", "request_id": request_id})
                terminal_sent = True
            return True

        def write_chunk(message: dict[str, Any]) -> bool:
            """Serialize cancellation against each chunk write."""

            with self._state_lock:
                if cancel.is_set():
                    return False
                self._write(message)
                return True

        try:
            prompt_audio = request.get("prompt_audio")
            prompt_text = request.get("prompt_text")
            if not prompt_audio or not prompt_text:
                raise ValueError("prompt_audio and prompt_text are required")
            if cancelled():
                return
            with redirect_stdout(sys.stderr):
                chunks = self._model.inference_zero_shot(
                    str(request["text"]), str(prompt_text), str(prompt_audio), stream=True
                )
                for index, item in enumerate(chunks):
                    if cancelled():
                        return
                    pcm = self._to_pcm16(item["tts_speech"])
                    if cancelled():
                        return
                    if not write_chunk(
                        {
                            "type": "chunk",
                            "request_id": request_id,
                            "chunk_id": f"cosyvoice-{request_id}-{index}",
                            "audio_b64": base64.b64encode(pcm).decode("ascii"),
                            "timestamp": time.time(),
                            "is_final": False,
                            **identity,
                        }
                    ):
                        cancelled()
                        return
            if cancelled():
                return
            with self._state_lock:
                if cancel.is_set():
                    emit_cancelled = True
                else:
                    self._write(
                        {
                            "type": "chunk",
                            "request_id": request_id,
                            "chunk_id": f"cosyvoice-{request_id}-final",
                            "audio_b64": "",
                            "timestamp": time.time(),
                            "is_final": True,
                            **identity,
                        }
                    )
                    self._write({"type": "done", "request_id": request_id})
                    terminal_sent = True
                    emit_cancelled = False
            if emit_cancelled:
                cancelled()
        except Exception as exc:  # pragma: no cover - exercised in real environment
            if not cancelled():
                self._write(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": f"{exc}\n{traceback.format_exc()}",
                    }
                )
        finally:
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._active = None
                    self._active_request_id = None
                    self._active_cancel = None

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
