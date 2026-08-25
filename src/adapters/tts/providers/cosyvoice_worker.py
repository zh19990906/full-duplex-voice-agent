"""Client and protocol helpers for an isolated CosyVoice worker process."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import AsyncIterable, Mapping
from pathlib import Path
from typing import Any, Sequence

from src.tts_runtime.stream import AudioChunk


def encode_worker_request(
    request_id: str,
    text: str,
    *,
    prompt_text: str | None = None,
    prompt_audio: str | None = None,
) -> str:
    """Encode one synthesis request as a newline-delimited JSON message."""

    message: dict[str, Any] = {
        "op": "synthesize",
        "request_id": request_id,
        "text": text,
    }
    if prompt_text is not None:
        message["prompt_text"] = prompt_text
    if prompt_audio is not None:
        message["prompt_audio"] = prompt_audio
    return json.dumps(message, ensure_ascii=False) + "\n"


def decode_worker_message(line: str) -> AudioChunk:
    """Decode a worker audio message into the project's audio contract."""

    message = json.loads(line)
    if message.get("type") != "chunk":
        raise ValueError(f"expected an audio chunk message, got {message.get('type')!r}")
    encoded = message.get("audio_b64", "")
    if not isinstance(encoded, str):
        raise TypeError("worker audio_b64 must be a string")
    return AudioChunk(
        str(message.get("chunk_id", "cosyvoice-audio")),
        base64.b64decode(encoded),
        float(message.get("timestamp", time.time())),
        bool(message.get("is_final", False)),
    )


class CosyVoiceWorkerClient:
    """Keep a CosyVoice process warm while exposing a provider-like runtime.

    The worker is intentionally a separate process because CosyVoice may use a
    different PyTorch/CUDA environment from the main voice-agent process.
    """

    def __init__(
        self,
        model_path: str,
        *,
        worker_command: Sequence[str] | str | None = None,
        worker_script: str | None = None,
        worker_python: str | None = None,
        cosyvoice_root: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        runtime: Any | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        if not model_path:
            raise ValueError("CosyVoice model_path must be supplied")
        self.model_path = model_path
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self.startup_timeout = startup_timeout
        self._runtime = runtime
        self._process: subprocess.Popen[str] | None = None
        self._write_lock = asyncio.Lock()
        self._command = self._build_command(
            model_path,
            worker_command=worker_command,
            worker_script=worker_script,
            worker_python=worker_python,
            cosyvoice_root=cosyvoice_root,
        )

    async def stream_audio(self, text: str, **options: Any) -> AsyncIterable[AudioChunk]:
        """Start synthesis and return an async iterator of PCM16 chunks."""

        if self._runtime is not None:
            result = self._runtime.stream_audio(text, **self._runtime_options(options))
            if hasattr(result, "__await__"):
                result = await result

            async def runtime_chunks() -> AsyncIterable[AudioChunk]:
                async for item in result:
                    yield self._normalize_runtime_item(item)

            return runtime_chunks()

        request_id = uuid.uuid4().hex
        await self._ensure_started()
        runtime_options = self._runtime_options(options)
        await self._send(
            encode_worker_request(
                request_id,
                text,
                prompt_text=runtime_options.get("prompt_text"),
                prompt_audio=runtime_options.get("prompt_audio"),
            )
        )

        async def worker_chunks() -> AsyncIterable[AudioChunk]:
            while True:
                line = await self._readline()
                message = json.loads(line)
                if message.get("request_id") != request_id:
                    raise RuntimeError("CosyVoice worker returned a mismatched request id")
                message_type = message.get("type")
                if message_type == "chunk":
                    yield decode_worker_message(line)
                elif message_type in {"done", "cancelled"}:
                    return
                elif message_type == "error":
                    raise RuntimeError(str(message.get("error", "CosyVoice worker failed")))
                else:
                    raise RuntimeError(f"unknown CosyVoice worker message: {message_type!r}")

        return worker_chunks()

    async def interrupt(self) -> None:
        """Request cancellation of the active worker synthesis."""

        if self._runtime is not None:
            method = getattr(self._runtime, "interrupt", None)
            if callable(method):
                result = method()
                if hasattr(result, "__await__"):
                    await result
            return
        if self._process is not None and self._process.poll() is None:
            await self._send(json.dumps({"op": "cancel"}) + "\n")

    async def close(self) -> None:
        """Stop the worker process if this client owns one."""

        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        await self._send_to_process(process, json.dumps({"op": "shutdown"}) + "\n")
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.terminate()
            await asyncio.to_thread(process.wait, 5)

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        try:
            line = await asyncio.wait_for(self._readline(), self.startup_timeout)
        except asyncio.TimeoutError as exc:
            await self._terminate_process()
            raise RuntimeError(
                f"CosyVoice worker did not become ready within {self.startup_timeout:.0f} seconds"
            ) from exc
        message = json.loads(line)
        if message.get("type") != "ready":
            await self._terminate_process()
            raise RuntimeError(f"CosyVoice worker failed to start: {message}")

    async def _send(self, line: str) -> None:
        if self._process is None:
            raise RuntimeError("CosyVoice worker is not running")
        await self._send_to_process(self._process, line)

    async def _send_to_process(self, process: subprocess.Popen[str], line: str) -> None:
        if process.stdin is None:
            raise RuntimeError("CosyVoice worker stdin is unavailable")
        async with self._write_lock:
            await asyncio.to_thread(process.stdin.write, line)
            await asyncio.to_thread(process.stdin.flush)

    async def _readline(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("CosyVoice worker stdout is unavailable")
        line = await asyncio.to_thread(self._process.stdout.readline)
        if not line:
            raise RuntimeError("CosyVoice worker exited unexpectedly")
        return line

    async def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)

    def _runtime_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(options)
        merged.setdefault("prompt_audio", self.prompt_audio)
        merged.setdefault("prompt_text", self.prompt_text)
        return {key: value for key, value in merged.items() if value is not None}

    @staticmethod
    def _normalize_runtime_item(item: Any) -> AudioChunk:
        if isinstance(item, AudioChunk):
            return item
        if isinstance(item, bytes):
            return AudioChunk("cosyvoice-audio", item, time.time(), False)
        if isinstance(item, Mapping):
            return AudioChunk(
                str(item.get("chunk_id", "cosyvoice-audio")),
                bytes(item.get("audio_data", item.get("data", b""))),
                float(item.get("timestamp", time.time())),
                bool(item.get("is_final", False)),
            )
        raise TypeError("CosyVoice worker runtime returned an unsupported item")

    @staticmethod
    def _build_command(
        model_path: str,
        *,
        worker_command: Sequence[str] | str | None,
        worker_script: str | None,
        worker_python: str | None,
        cosyvoice_root: str | None,
    ) -> list[str]:
        if worker_command is not None:
            return shlex.split(worker_command) if isinstance(worker_command, str) else list(worker_command)
        script = worker_script or str(Path(__file__).resolve().parents[4] / "scripts" / "cosyvoice_worker.py")
        python = worker_python or "python3"
        command = [python, script, "--model", model_path]
        if cosyvoice_root:
            command.extend(["--cosy-root", cosyvoice_root])
        return command


__all__ = [
    "CosyVoiceWorkerClient",
    "decode_worker_message",
    "encode_worker_request",
]
