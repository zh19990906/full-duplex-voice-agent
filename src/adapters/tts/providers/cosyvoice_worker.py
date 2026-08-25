"""Client and JSONL protocol helpers for an isolated CosyVoice worker."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, replace
import inspect
import json
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import AsyncIterable, Mapping
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.tts_runtime.stream import AudioChunk


def _require_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a nonempty string")
    return request_id


def _validate_identity(
    response_id: Any = None, generation_epoch: Any = None, segment_id: Any = None
) -> tuple[str | None, int | None, int | None]:
    if response_id is not None and (not isinstance(response_id, str) or not response_id):
        raise ValueError("response_id must be a nonempty string or None")
    for name, value in (("generation_epoch", generation_epoch), ("segment_id", segment_id)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer or None")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
    return response_id, generation_epoch, segment_id


def encode_worker_request(
    request_id: str,
    text: str,
    *,
    prompt_text: str | None = None,
    prompt_audio: str | None = None,
    response_id: str | None = None,
    generation_epoch: int | None = None,
    segment_id: int | None = None,
) -> str:
    """Encode a version-stable identity-bearing synthesize request as JSONL."""

    _require_request_id(request_id)
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    response_id, generation_epoch, segment_id = _validate_identity(
        response_id, generation_epoch, segment_id
    )
    message: dict[str, Any] = {"op": "synthesize", "request_id": request_id, "text": text}
    for name, value in (
        ("prompt_text", prompt_text),
        ("prompt_audio", prompt_audio),
        ("response_id", response_id),
        ("generation_epoch", generation_epoch),
        ("segment_id", segment_id),
    ):
        if value is not None:
            message[name] = value
    return json.dumps(message, ensure_ascii=False) + "\n"


def encode_worker_cancel(request_id: str) -> str:
    """Encode a request-scoped cancellation command as JSONL."""

    return json.dumps({"op": "cancel", "request_id": _require_request_id(request_id)}) + "\n"


def encode_worker_shutdown() -> str:
    """Encode the version-stable worker shutdown command."""

    return json.dumps({"op": "shutdown"}) + "\n"


def _is_audio_message(message: Mapping[str, Any]) -> bool:
    return message.get("type") == "chunk" or message.get("event") in {"audio", "pcm"}


def decode_worker_message(
    line: str,
    *,
    response_id: str | None = None,
    generation_epoch: int | None = None,
    segment_id: int | None = None,
) -> AudioChunk:
    """Decode either supported worker audio spelling into ``AudioChunk``."""

    message = json.loads(line)
    if not isinstance(message, Mapping) or not _is_audio_message(message):
        event = message.get("type") if isinstance(message, Mapping) else type(message).__name__
        raise ValueError(f"expected an audio chunk message, got {event!r}")
    request_id = _require_request_id(message.get("request_id"))
    encoded = message.get("audio_b64", message.get("pcm"))
    if not isinstance(encoded, str):
        raise TypeError("worker audio payload must be a base64 string")
    try:
        audio_data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("worker audio payload is not valid base64") from exc
    response_id, generation_epoch, segment_id = _validate_identity(
        message.get("response_id", response_id),
        message.get("generation_epoch", generation_epoch),
        message.get("segment_id", segment_id),
    )
    return AudioChunk(
        str(message.get("chunk_id", "cosyvoice-audio")),
        audio_data,
        float(message.get("timestamp", time.time())),
        bool(message.get("is_final", False)),
        request_id,
        response_id,
        generation_epoch,
        segment_id,
    )


class _WorkerTransport(Protocol):
    """Private injectable transport. Client owns its single stdout reader."""

    async def start(self) -> None: ...

    async def send(self, line: str) -> None: ...

    async def readline(self) -> str: ...

    async def close(self) -> None: ...


class _SubprocessWorkerTransport:
    """``_WorkerTransport`` adapter around the CosyVoice subprocess."""

    def __init__(self, command: Sequence[str]) -> None:
        self._command = list(command)
        self._process: subprocess.Popen[str] | None = None

    async def start(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )

    async def send(self, line: str) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("CosyVoice worker stdin is unavailable")
        await asyncio.to_thread(process.stdin.write, line)
        await asyncio.to_thread(process.stdin.flush)

    async def readline(self) -> str:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("CosyVoice worker stdout is unavailable")
        line = await asyncio.to_thread(process.stdout.readline)
        if not line:
            raise RuntimeError("CosyVoice worker exited unexpectedly")
        return line

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, 5)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                await asyncio.to_thread(stream.close)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("CosyVoice worker is not running")
        return self._process


@dataclass
class _RequestState:
    request_id: str
    response_id: str | None
    generation_epoch: int | None
    segment_id: int | None
    cancelled: bool = False


class CosyVoiceWorkerClient:
    """A serialized, request-identified CosyVoice worker client."""

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
        transport: _WorkerTransport | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        if not model_path:
            raise ValueError("CosyVoice model_path must be supplied")
        if runtime is not None and transport is not None:
            raise ValueError("runtime and transport are mutually exclusive")
        self.model_path = model_path
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self.startup_timeout = startup_timeout
        self._runtime = runtime
        self._transport = transport
        self._injected_transport = transport is not None
        self._started = False
        self._closed = False
        self._broken = False
        self._active: _RequestState | None = None
        self._write_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._command = self._build_command(
            model_path,
            worker_command=worker_command,
            worker_script=worker_script,
            worker_python=worker_python,
            cosyvoice_root=cosyvoice_root,
        )

    async def stream_audio(
        self,
        text: str,
        *,
        request_id: str | None = None,
        response_id: str | None = None,
        generation_epoch: int | None = None,
        segment_id: int | None = None,
        **options: Any,
    ) -> AsyncIterable[AudioChunk]:
        """Start one request. A second stream is rejected until terminal drain."""

        request_id = _require_request_id(request_id or uuid.uuid4().hex)
        response_id, generation_epoch, segment_id = _validate_identity(
            response_id, generation_epoch, segment_id
        )
        state = _RequestState(request_id, response_id, generation_epoch, segment_id)
        await self._claim_request(state)
        try:
            if self._runtime is not None:
                result = self._call_runtime_stream(text, state, options)
                if inspect.isawaitable(result):
                    result = await result
                return self._runtime_chunks(result, state)
            await self._ensure_started()
            await self._send(
                encode_worker_request(
                    request_id,
                    text,
                    prompt_text=self._runtime_options(options).get("prompt_text"),
                    prompt_audio=self._runtime_options(options).get("prompt_audio"),
                    response_id=response_id,
                    generation_epoch=generation_epoch,
                    segment_id=segment_id,
                )
            )
            if state.cancelled:
                await self._send(encode_worker_cancel(request_id))
            return self._worker_chunks(state)
        except BaseException:
            await self._clear_request(state)
            raise

    async def cancel(self, request_id: str) -> None:
        """Mark cancellation before the serialized write; repeating it is safe."""

        request_id = _require_request_id(request_id)
        async with self._request_lock:
            state = self._active
            if state is None or state.request_id != request_id:
                raise ValueError("request_id does not match the active CosyVoice request")
            if state.cancelled:
                return
            state.cancelled = True
        if self._runtime is not None:
            await self._call_runtime_cancel(request_id)
        elif self._started and not self._closed:
            await self._send(encode_worker_cancel(request_id))

    async def interrupt(self) -> None:
        """Cancel the active worker request with its known request id."""

        async with self._request_lock:
            request_id = self._active.request_id if self._active is not None else None
        if request_id is not None:
            await self.cancel(request_id)
        elif self._runtime is not None:
            await self._call_runtime_cancel(None)

    async def close(self) -> None:
        """Resolve active work then send shutdown through a captured transport."""

        async with self._request_lock:
            active_id = self._active.request_id if self._active is not None else None
        if active_id is not None:
            try:
                await self.cancel(active_id)
            except (RuntimeError, ValueError):
                pass
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            transport = self._transport
            self._transport = None
            self._started = False
            if self._runtime is not None:
                closer = getattr(self._runtime, "close", None)
                if callable(closer):
                    result = closer()
                    if inspect.isawaitable(result):
                        await result
            elif transport is not None:
                try:
                    await self._send_to(transport, encode_worker_shutdown())
                finally:
                    await transport.close()
        async with self._request_lock:
            self._active = None

    async def _claim_request(self, state: _RequestState) -> None:
        async with self._request_lock:
            if self._closed:
                raise RuntimeError("CosyVoice worker client is closed")
            if self._broken:
                raise RuntimeError("CosyVoice worker transport is broken")
            if self._active is not None:
                raise RuntimeError("a CosyVoice synthesis request is already active")
            self._active = state

    async def _clear_request(self, state: _RequestState) -> None:
        async with self._request_lock:
            if self._active is state:
                self._active = None

    async def _ensure_started(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("CosyVoice worker client is closed")
            if self._broken:
                raise RuntimeError("CosyVoice worker transport is broken")
            if self._started:
                return
            if self._transport is None:
                self._transport = _SubprocessWorkerTransport(self._command)
            await self._transport.start()
            if not self._injected_transport:
                try:
                    line = await asyncio.wait_for(self._transport.readline(), self.startup_timeout)
                except asyncio.TimeoutError as exc:
                    await self._fail_start_locked()
                    raise RuntimeError(
                        f"CosyVoice worker did not become ready within {self.startup_timeout:.0f} seconds"
                    ) from exc
                try:
                    ready = json.loads(line)
                except json.JSONDecodeError as exc:
                    await self._fail_start_locked()
                    raise RuntimeError("CosyVoice worker emitted malformed startup JSON") from exc
                if ready.get("type") != "ready":
                    await self._fail_start_locked()
                    raise RuntimeError(f"CosyVoice worker failed to start: {ready}")
            self._started = True

    async def _fail_start_locked(self) -> None:
        """Close failed startup while the lifecycle lock is already held."""

        self._broken = True
        transport = self._transport
        self._transport = None
        self._started = False
        if transport is not None:
            await transport.close()

    async def _send(self, line: str) -> None:
        transport = self._transport
        if transport is None:
            raise RuntimeError("CosyVoice worker is not running")
        await self._send_to(transport, line)

    async def _send_to(self, transport: _WorkerTransport, line: str) -> None:
        async with self._write_lock:
            await transport.send(line)

    async def _worker_chunks(self, state: _RequestState) -> AsyncIterable[AudioChunk]:
        terminal = False
        try:
            while True:
                transport = self._transport
                if transport is None:
                    raise RuntimeError("CosyVoice worker transport is unavailable")
                try:
                    line = await transport.readline()
                    message = json.loads(line)
                except (json.JSONDecodeError, RuntimeError) as exc:
                    await self._break_transport()
                    raise RuntimeError("CosyVoice worker protocol ended unexpectedly") from exc
                if not isinstance(message, Mapping) or message.get("request_id") != state.request_id:
                    await self._break_transport()
                    raise RuntimeError("CosyVoice worker returned a mismatched request id")
                event = message.get("type", message.get("event"))
                if _is_audio_message(message):
                    try:
                        chunk = decode_worker_message(
                            line,
                            response_id=state.response_id,
                            generation_epoch=state.generation_epoch,
                            segment_id=state.segment_id,
                        )
                    except (TypeError, ValueError) as exc:
                        await self._break_transport()
                        raise RuntimeError("CosyVoice worker emitted malformed audio") from exc
                    if not state.cancelled:
                        yield chunk
                    continue
                if event in {"done", "cancelled"}:
                    terminal = True
                    await self._clear_request(state)
                    return
                if event == "error":
                    if state.cancelled:
                        continue
                    await self._clear_request(state)
                    raise RuntimeError(str(message.get("error", "CosyVoice worker failed")))
                await self._break_transport()
                raise RuntimeError(f"unknown CosyVoice worker message: {event!r}")
        finally:
            if not terminal and self._active is state:
                try:
                    await self.cancel(state.request_id)
                except (RuntimeError, ValueError):
                    pass

    async def _runtime_chunks(self, result: Any, state: _RequestState) -> AsyncIterable[AudioChunk]:
        terminal = False
        try:
            async for item in result:
                if not state.cancelled:
                    yield self._normalize_runtime_item(item, state)
            terminal = True
            await self._clear_request(state)
        finally:
            if not terminal and self._active is state:
                try:
                    await self.cancel(state.request_id)
                except (RuntimeError, ValueError):
                    pass

    async def _break_transport(self) -> None:
        async with self._lifecycle_lock:
            self._broken = True
            transport = self._transport
            self._transport = None
            self._started = False
        if transport is not None:
            await transport.close()

    def _call_runtime_stream(self, text: str, state: _RequestState, options: Mapping[str, Any]) -> Any:
        kwargs = self._runtime_options(options)
        kwargs.update(
            request_id=state.request_id,
            response_id=state.response_id,
            generation_epoch=state.generation_epoch,
            segment_id=state.segment_id,
        )
        return self._call_supported(self._runtime.stream_audio, text, **kwargs)

    async def _call_runtime_cancel(self, request_id: str | None) -> None:
        method = getattr(self._runtime, "cancel", None) or getattr(self._runtime, "interrupt", None)
        if not callable(method):
            return
        result = method() if request_id is None else self._call_supported(method, request_id)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _call_supported(method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return method(*args, **kwargs)
        positional_count = sum(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        return method(
            *args[:positional_count],
            **{key: value for key, value in kwargs.items() if key in signature.parameters},
        )

    def _runtime_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(options)
        merged.setdefault("prompt_audio", self.prompt_audio)
        merged.setdefault("prompt_text", self.prompt_text)
        return {key: value for key, value in merged.items() if value is not None}

    @staticmethod
    def _normalize_runtime_item(item: Any, state: _RequestState) -> AudioChunk:
        if isinstance(item, AudioChunk):
            for name, expected in (
                ("request_id", state.request_id),
                ("response_id", state.response_id),
                ("generation_epoch", state.generation_epoch),
                ("segment_id", state.segment_id),
            ):
                actual = getattr(item, name)
                if actual is not None and actual != expected:
                    raise RuntimeError(f"CosyVoice runtime returned a mismatched {name}")
            return replace(
                item,
                request_id=state.request_id,
                response_id=state.response_id,
                generation_epoch=state.generation_epoch,
                segment_id=state.segment_id,
            )
        if isinstance(item, bytes):
            return AudioChunk(
                "cosyvoice-audio", item, time.time(), False,
                state.request_id, state.response_id, state.generation_epoch, state.segment_id,
            )
        if isinstance(item, Mapping):
            return AudioChunk(
                str(item.get("chunk_id", "cosyvoice-audio")),
                bytes(item.get("audio_data", item.get("data", b""))),
                float(item.get("timestamp", time.time())),
                bool(item.get("is_final", False)),
                state.request_id, state.response_id, state.generation_epoch, state.segment_id,
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
    "encode_worker_cancel",
    "encode_worker_request",
    "encode_worker_shutdown",
]
