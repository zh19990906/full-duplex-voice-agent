"""Production application entrypoint and graceful shutdown coordinator."""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from src.config import ConfigLoader
from src.model_runtime.lifecycle import ModelLifecycleManager, ModelState
from src.ops.health import HealthChecker
from src.ops.logging_config import configure_logging, log_event
from src.ops.status import RuntimeStatus
from src.runtime_app.container import ApplicationContainer


AsyncHook = Callable[[], Awaitable[None] | None]


class ProductionRuntime:
    """Compose existing application/lifecycle components for operations."""

    def __init__(
        self,
        profile: str | None = None,
        config_loader: ConfigLoader | Any | None = None,
        application: ApplicationContainer | None = None,
        lifecycle_manager: ModelLifecycleManager | None = None,
        logger: Any | None = None,
        tts_stop: AsyncHook | None = None,
        model_resources: Mapping[str, Any] | None = None,
    ) -> None:
        self.profile = profile or os.environ.get("VOICE_AGENT_PROFILE", "dev")
        self.config_loader = config_loader or ConfigLoader()
        self.application = application or ApplicationContainer()
        self.lifecycle = lifecycle_manager or ModelLifecycleManager()
        self.logger = logger or configure_logging()
        self.tts_stop = tts_stop
        self.model_resources = dict(model_resources or {})
        self.config: Mapping[str, Any] | None = None
        self.started_at: float | None = None
        self.shutdown_requested = False
        self._shutdown_complete = False
        self._shutdown_event: asyncio.Event | None = None

    async def initialize(self) -> None:
        """Load configuration, prepare configured models, then accept events."""

        if self.application.initialized:
            return
        self.config = self.config_loader.load(self.profile)
        self._register_models(self.config)
        await self._load_auto_models()
        await self.application.initialize()
        self.started_at = time.monotonic()
        log_event(self.logger, "runtime_started", profile=self.profile)

    async def shutdown(self) -> None:
        """Stop intake, cancel generation, stop output, and unload resources."""

        if self._shutdown_complete:
            return
        self.shutdown_requested = True
        log_event(self.logger, "shutdown_started", profile=self.profile)
        try:
            await self.application.voice_agent.stop()
            if self.application.generation_manager.active_session is not None:
                await self.application.generation_manager.cancel_current()
                log_event(self.logger, "generation_cancelled", reason="shutdown")
            if self.tts_stop is not None:
                result = self.tts_stop()
                if inspect.isawaitable(result):
                    await result
            await self.application.shutdown()
            await self._unload_models()
        finally:
            self._shutdown_complete = True
            log_event(self.logger, "runtime_stopped", profile=self.profile)

    def request_shutdown(self, *_signals: Any) -> None:
        """Record a signal request without doing async work in the handler."""

        self.shutdown_requested = True
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()
        log_event(self.logger, "shutdown_requested")

    async def wait_for_shutdown(self) -> None:
        """Wait for SIGINT/SIGTERM request and perform graceful shutdown."""

        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        await self._shutdown_event.wait()
        await self.shutdown()

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install portable SIGINT/SIGTERM handlers where supported."""

        event_loop = loop or asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                event_loop.add_signal_handler(signum, self.request_shutdown, signum)
            except (NotImplementedError, RuntimeError):
                signal.signal(signum, self.request_shutdown)

    def health(self) -> dict[str, Any]:
        return HealthChecker(self.application, self.lifecycle).health()

    def status(self) -> RuntimeStatus:
        loaded_models: dict[str, str] = {}
        for name in ("asr", "llm", "tts"):
            try:
                loaded_models[name] = self.lifecycle.status(name).value.lower()
            except KeyError:
                loaded_models[name] = "unknown"
        return RuntimeStatus(
            profile=self.profile,
            runtime_state="running" if self.application.initialized else "stopped",
            loaded_models=loaded_models,
            started_at=self.started_at,
        )

    def _register_models(self, config: Mapping[str, Any]) -> None:
        models = config.get("models", {})
        for name in ("asr", "llm", "tts"):
            settings = models[name]
            self.lifecycle.register(
                name=name,
                provider=str(settings["provider"]),
                model_path=str(settings.get("model_path", settings.get("local_path", f"models/{name}"))),
                resource=self.model_resources.get(name),
                auto_load=bool(settings.get("auto_load", False)),
            )

    async def _load_auto_models(self) -> None:
        for name in ("asr", "llm", "tts"):
            record = self.lifecycle.record(name)
            if not record.auto_load or record.resource is None:
                continue
            await self.lifecycle.load(name)
            await self.lifecycle.warmup(name)
            log_event(self.logger, "model_ready", model=name, status=record.status.value.lower())

    async def _unload_models(self) -> None:
        for name in reversed(("asr", "llm", "tts")):
            record = self.lifecycle.record(name)
            if record.state not in {ModelState.READY, ModelState.ACTIVE}:
                continue
            await self.lifecycle.unload(name)
            log_event(self.logger, "model_unloaded", model=name, status=record.status.value.lower())


async def run_async(profile: str | None = None) -> None:
    runtime = ProductionRuntime(profile=profile)
    await runtime.initialize()
    runtime.install_signal_handlers()
    await runtime.wait_for_shutdown()


def run(profile: str | None = None) -> None:
    """Run the production process until SIGINT or SIGTERM."""

    asyncio.run(run_async(profile))


if __name__ == "__main__":
    run()
